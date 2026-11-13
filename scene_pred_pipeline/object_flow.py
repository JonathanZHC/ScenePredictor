"""How the tracked objects are distributed over DifFlow3D forwards (difflow ``object_flow.mode``).

joint       ``DifFlowPredictor``: one forward for the joint cloud of all objects. The cheapest. The pyramid's coarse
            levels are sampled by spatial extent, so a thin or small object next to a large one keeps a handful of
            coarse points (a 0.6 m stick held by a person: 21 of 256, 6 of 64); its extremity then gets a biased flow
            and a large uncertainty.
per_object  ``PerObjectFlowPredictor`` (this module): every object a forward of its own.
scheduled   ``ScheduledFlowPredictor`` (scheduled_flow.py): forwards of their own only where they pay, rows of a
            batched forward for everything else.

What the last two share lives here:
  ObjectLayout   where the objects of a frame are on the context grid. An object's cloud is all its points plus, as
                 context, the other objects' points in the cells (edge ``context_cell_m``) that it occupies or that
                 touch one it occupies. Objects in contact need each other (the hand that holds a stick anchors the
                 stick's motion); without context the fast-motion error grows, with too much the small object is
                 crowded out again. Found with one scatter and one 3x3x3 max-pool for all objects - no distance
                 matrix. The context depends on the frame alone, so a frame's cloud is the same as the target of one
                 pair and the source of the next and the runner's encoding reuse stays valid.
  ForwardPool    a fixed number of ``DifFlowPredictor``s (the deployed runner each: CUDA graphs, point buckets,
                 normalisation, encoding reuse), every one on its own CUDA stream, so their forwards overlap on the
                 GPU (RTX 5090: about 3.6 ms + 0.67 ms per 1000 points + 0.9 ms per further forward).
  AnchorParts    collects the anchors of the forwards into one ``FlowResult``. Only an object's OWN anchors are kept
                 (the same points estimated as context in a neighbour's forward are worse), so dense recovery stays
                 same-track.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Callable, Iterable

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair
from .flow_prediction import DifFlowPredictor

# Context grid: cells per axis, counted from the cloud's minimum (0.15 m cells: 6.0 x 6.0 x 4.2 m).
GRID_CELLS = (40, 40, 28)


def build_flow_predictor(config: PipelineConfig, *, upstream_voxel_size_m: float):
    """The flow stage for ``config.difflow.object_flow.mode``; a single trackable instance is always the joint forward."""
    object_flow = config.difflow.object_flow
    instances = config.tracker.max_tracked_instances
    if object_flow.mode == "joint" or instances <= 1:
        return DifFlowPredictor(config, upstream_voxel_size_m=upstream_voxel_size_m)
    if object_flow.mode == "per_object":
        return PerObjectFlowPredictor(
            config, upstream_voxel_size_m=upstream_voxel_size_m, max_objects=min(int(object_flow.max_objects), instances)
        )
    from .scheduled_flow import ScheduledFlowPredictor

    return ScheduledFlowPredictor(config, upstream_voxel_size_m=upstream_voxel_size_m)


# ----------------------------------------------------------------------------------------------------------------------
# Geometry of the objects of one frame
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class ObjectLayout:
    """The objects ``object_ids`` (sorted track ids, K of them) in one frame's cloud. Device tensors, no host read."""

    points: torch.Tensor         # [N, 3] the frame's cloud
    row: torch.Tensor            # [N] index into object_ids of every point's object, K for points of other tracks
    count: torch.Tensor          # [K] points per object
    extent: torch.Tensor         # [K] largest side of the bounding box (m), 0 without points
    cell: torch.Tensor | None    # [N] flat grid cell of every point
    occupied: torch.Tensor | None    # [K, C] cells that hold a point of the object
    reach: torch.Tensor | None       # [K, C] those cells and the cells next to them
    _by_object: tuple[torch.Tensor, torch.Tensor] | None = None    # point order sorted by object, every object's first position

    @classmethod
    def of(cls, points: torch.Tensor, track_ids: torch.Tensor, object_ids: torch.Tensor, cell_m: float, *, grid: bool = True):
        objects = int(object_ids.numel())
        row = torch.searchsorted(object_ids, track_ids).clamp_(max=objects - 1)
        row = torch.where(object_ids[row] == track_ids, row, torch.full_like(row, objects))
        count = torch.zeros(objects + 1, device=points.device).index_add_(0, row, torch.ones_like(row, dtype=torch.float32))[:objects]
        spread = row[:, None].expand(-1, 3)
        low = torch.full((objects + 1, 3), 1.0e9, device=points.device, dtype=points.dtype).scatter_reduce_(0, spread, points, "amin")
        high = torch.full((objects + 1, 3), -1.0e9, device=points.device, dtype=points.dtype).scatter_reduce_(0, spread, points, "amax")
        extent = (high[:objects] - low[:objects]).amax(1).clamp_min(0.0).float()
        if not grid:
            return cls(points, row, count, extent, None, None, None)
        dims = torch.tensor(GRID_CELLS, device=points.device)
        index = torch.minimum(torch.floor((points - (points.min(0).values - cell_m)) / cell_m).long().clamp_(min=0), dims - 1)
        cell = (index[:, 0] * GRID_CELLS[1] + index[:, 1]) * GRID_CELLS[2] + index[:, 2]
        cells = GRID_CELLS[0] * GRID_CELLS[1] * GRID_CELLS[2]
        occupied = torch.zeros(objects + 1, cells, device=points.device)
        occupied[row, cell] = 1.0
        occupied = occupied[:objects]
        reach = torch.nn.functional.max_pool3d(occupied.view(1, objects, *GRID_CELLS), 3, 1, 1).view(objects, cells)
        return cls(points, row, count, extent, cell, occupied, reach)

    def cloud_mask(self, rows: torch.Tensor) -> torch.Tensor:
        """[N] the points of the objects ``rows`` and, as context, the other points in or next to their cells."""
        return self.reach[rows].amax(0)[self.cell] > 0

    def touching(self) -> torch.Tensor:
        """[K, K] entry (k, j): object j has a point in or next to a cell of object k."""
        return (self.reach @ self.occupied.T) > 0

    def evenly_spaced(self, rows: list[int], number: int) -> torch.Tensor:
        """[R, number, 3] points of each of the objects ``rows``, evenly spaced in the cloud's own order (repeated if
        an object has fewer)."""
        if self._by_object is None:
            self._by_object = (torch.argsort(self.row, stable=True), (torch.cumsum(self.count, 0) - self.count).long())
        order, first = self._by_object
        rows = torch.tensor(rows, device=self.points.device)
        steps = torch.linspace(0.0, 1.0, number, device=self.points.device)
        position = first[rows][:, None] + (steps[None] * (self.count[rows] - 1.0)[:, None]).round().long()
        return self.points[order[position]].float()


# ----------------------------------------------------------------------------------------------------------------------
# Forwards side by side
# ----------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ForwardJob:
    """One forward: the cloud of ``current_mask`` on the pool's ``slot``; the anchors of the tracks ``own`` are kept."""

    slot: int
    own: tuple[int, ...]
    signature: tuple               # who is in the cloud and how it is thinned: the slot's encoding-reuse key
    current_mask: torch.Tensor     # [N current points]
    previous_mask: torch.Tensor | None    # None: the slot's runner still holds the previous frame's encoding


class SlotMap:
    """Keeps a key (an object, or the shared forward) on the slot it had: that slot's runner holds its previous frame."""

    def __init__(self, size: int) -> None:
        self.size = int(size)
        self._slot: dict[object, int] = {}

    def assign(self, keys: Iterable[object]) -> dict[object, int]:
        keys = list(keys)
        self._slot = {key: slot for key, slot in self._slot.items() if key in keys}
        for key in keys:
            if key not in self._slot:
                used = set(self._slot.values())
                self._slot[key] = next(slot for slot in range(self.size) if slot not in used)
        return dict(self._slot)

    def clear(self) -> None:
        self._slot.clear()


class ForwardPool:
    """``size`` deployed predictors, each on its own CUDA stream of the pipeline's (highest) priority."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        upstream_voxel_size_m: float,
        size: int,
        predictor_factory: Callable[[], object] | None = None,
    ) -> None:
        if size < 1:
            raise ValueError("a forward pool needs at least one predictor")
        self.device = torch.device(config.runtime.device)
        factory = predictor_factory or (lambda: DifFlowPredictor(config, upstream_voxel_size_m=upstream_voxel_size_m))
        self.predictors = [factory() for _ in range(int(size))]
        self._streams: list[torch.cuda.Stream] | None = None

    def __len__(self) -> int:
        return len(self.predictors)

    @property
    def first(self):
        return self.predictors[0]

    def prepare(self) -> None:
        """Capture every predictor's CUDA graphs (before the tracker's graphs exist) and create the streams."""
        for predictor in self.predictors:
            predictor.prepare()
        if self.device.type == "cuda":
            # the asynchronous SAM3 refresh must not delay a forward (cuda_streams.py)
            _, greatest = torch.cuda.Stream.priority_range()
            self._streams = [torch.cuda.Stream(device=self.device, priority=int(greatest)) for _ in self.predictors]

    def reset(self) -> None:
        for predictor in self.predictors:
            predictor.reset()

    def reset_calibration(self) -> None:
        for predictor in self.predictors:
            predictor.reset_calibration()

    def holds_previous(self, slot: int, pair: InstancePair, signature: tuple) -> bool:
        """The slot's runner encoded exactly this cloud of the pair's previous frame in the last cycle."""
        predictor = self.predictors[slot]
        return (
            getattr(predictor, "_cached_target_stamp_ns", None) == int(pair.previous_stamp_ns)
            and getattr(predictor, "_cached_track_signature", None) == signature
        )

    def launch(self, pair: InstancePair, jobs: list[ForwardJob], track_labels) -> dict[int, tuple[FlowResult, InstancePair]]:
        """Start every job's forward on its slot's stream; ``join`` waits for them."""
        if self._streams is not None and jobs:
            torch.cuda.current_stream(self.device).synchronize()      # the masks are complete before another stream reads them
        results = {}
        for job in jobs:
            stream = contextlib.nullcontext() if self._streams is None else torch.cuda.stream(self._streams[job.slot])
            with stream:
                held = job.previous_mask is None
                sub = InstancePair(
                    previous_stamp_ns=pair.previous_stamp_ns,
                    current_stamp_ns=pair.current_stamp_ns,
                    common_track_ids=job.signature,
                    previous_points=pair.previous_points if held else pair.previous_points[job.previous_mask],
                    current_points=pair.current_points[job.current_mask],
                    previous_track_ids=pair.previous_track_ids if held else pair.previous_track_ids[job.previous_mask],
                    current_track_ids=pair.current_track_ids[job.current_mask],
                    dt_s=pair.dt_s,
                )
                results[job.slot] = (self.predictors[job.slot].predict(sub, track_labels=track_labels), sub)
        return results

    def join(self, jobs: list[ForwardJob]) -> None:
        if self._streams is not None:
            for job in jobs:
                self._streams[job.slot].synchronize()

    def resolve(self, slots: list[int]) -> tuple[dict[str, float], dict[str, object], dict[str, object]]:
        """Timing and outlier statistics of the slots that ran. The forwards overlap, so a stage's time is the largest
        over the slots; voxel counts add up."""
        timing: dict[str, float] = {}
        info: dict[str, object] = {}
        debug: dict[str, object] = {}
        for slot in slots:
            slot_timing, slot_info, slot_debug = self.predictors[slot].resolve_after_cycle_fence()
            for name, value in slot_timing.items():
                timing[name] = max(timing.get(name, 0.0), float(value))
            if not info:
                info = dict(slot_info)
            else:
                info["applied"] = bool(info.get("applied", False)) or bool(slot_info.get("applied", False))
                for name in ("input_voxels", "retained_voxels", "removed_voxels"):
                    info[name] = int(info.get(name, 0)) + int(slot_info.get(name, 0))
            if slot_debug and not debug:
                debug = slot_debug
        if slots:
            timing["runner_objects"] = float(len(slots))
        return timing, info, debug


class AnchorParts:
    """The anchors of several forwards, concatenated into one ``FlowResult``."""

    def __init__(self) -> None:
        self._parts: dict[str, list[torch.Tensor]] = {name: [] for name in ("source", "warped", "flow", "ids", "cov", "scale")}

    def __bool__(self) -> bool:
        return bool(self._parts["source"])

    def add(self, source, warped, flow, ids, cov=None, scale=None) -> None:
        for name, value in (("source", source), ("warped", warped), ("flow", flow), ("ids", ids)):
            self._parts[name].append(value)
        if cov is not None:
            self._parts["cov"].append(cov.float())
            self._parts["scale"].append(scale)

    def add_own(self, result: FlowResult, own: torch.Tensor) -> None:
        """The anchors of ``result`` that belong to the tracks ``own``."""
        keep = torch.isin(result.source_anchor_track_ids, own)
        has_cov = result.anchor_flow_cov is not None
        self.add(
            result.source_anchors[keep], result.warped_anchors[keep], result.anchor_flow[keep],
            result.source_anchor_track_ids[keep],
            result.anchor_flow_cov[keep] if has_cov else None, result.anchor_flow_cov_scale[keep] if has_cov else None,
        )

    def flow_result(self, keep_mask: torch.Tensor, dtype: torch.dtype) -> FlowResult:
        parts = self._parts
        return FlowResult(
            source_anchors=torch.cat(parts["source"]).to(dtype),
            warped_anchors=torch.cat(parts["warped"]).to(dtype),
            anchor_flow=torch.cat(parts["flow"]).to(dtype),
            source_anchor_track_ids=torch.cat(parts["ids"]),
            target_input_keep_mask=keep_mask,
            anchor_flow_cov=torch.cat(parts["cov"]) if parts["cov"] else None,
            anchor_flow_cov_scale=torch.cat(parts["scale"]) if parts["scale"] else None,
        )


def keep_mask_of(pair: InstancePair, jobs: list[ForwardJob], results: dict[int, tuple[FlowResult, InstancePair]]) -> torch.Tensor:
    """``target_input_keep_mask`` over the pair's current points: what each forward's outlier filter kept of its own points."""
    keep = torch.ones(pair.current_points.shape[0], dtype=torch.bool, device=pair.current_points.device)
    for job in jobs:
        result, sub = results[job.slot]
        own = torch.isin(sub.current_track_ids, torch.tensor(job.own, device=sub.current_track_ids.device, dtype=sub.current_track_ids.dtype))
        keep[job.current_mask.nonzero(as_tuple=True)[0][own]] = result.target_input_keep_mask[own]
    return keep


class _PoolBacked:
    """The ``DifFlowPredictor`` surface the pipeline uses, for a flow stage that runs on a ``ForwardPool``."""

    pool: ForwardPool
    _ran: list[int]

    @property
    def calibration_scale(self):
        return self.pool.first.calibration_scale

    @property
    def outlier_filter_enabled(self) -> bool:
        return self.pool.first.outlier_filter_enabled

    @property
    def detailed_outlier_output(self) -> bool:
        return self.pool.first.detailed_outlier_output

    def prepare(self) -> None:
        self.pool.prepare()

    def reset(self) -> None:
        self.pool.reset()
        self._forget()

    def reset_calibration(self) -> None:
        self.pool.reset_calibration()
        self._forget()

    def resolve_after_cycle_fence(self):
        return self.pool.resolve(self._ran)

    def _forget(self) -> None:
        self._ran = []


# ----------------------------------------------------------------------------------------------------------------------
# per_object
# ----------------------------------------------------------------------------------------------------------------------
class PerObjectFlowPredictor(_PoolBacked):
    """Every object a forward of its own with its neighbours as context. Objects beyond the pool's size and objects with
    fewer than ``min_points`` points ride in the largest object's cloud (joint with it, their anchors kept)."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        upstream_voxel_size_m: float,
        max_objects: int,
        predictor_factory: Callable[[], object] | None = None,
    ) -> None:
        object_flow = config.difflow.object_flow
        self.context_cell_m = float(object_flow.context_cell_m)
        self.min_points = int(object_flow.min_points)
        self.pool = ForwardPool(config, upstream_voxel_size_m=upstream_voxel_size_m, size=max_objects, predictor_factory=predictor_factory)
        self._slots = SlotMap(len(self.pool))
        self._ran: list[int] = []

    def _forget(self) -> None:
        self._ran = []
        self._slots.clear()

    def predict(self, pair: InstancePair, *, track_labels: dict[int, str] | None = None) -> FlowResult:
        objects = sorted(int(track) for track in pair.common_track_ids)
        if not objects:
            self._ran = [0]
            return self.pool.first.predict(pair, track_labels=track_labels)
        with torch.inference_mode():
            object_ids = torch.tensor(objects, device=pair.current_track_ids.device, dtype=pair.current_track_ids.dtype)
            several = len(objects) > 1
            current = ObjectLayout.of(pair.current_points, pair.current_track_ids, object_ids, self.context_cell_m, grid=several)
            previous = ObjectLayout.of(pair.previous_points, pair.previous_track_ids, object_ids, self.context_cell_m, grid=False)
            count = dict(zip(objects, torch.minimum(current.count, previous.count).tolist()))     # the planning step's host read

            ranked = sorted(objects, key=lambda track: -count[track])
            heads = [track for track in ranked[: len(self.pool)] if count[track] >= self.min_points] or ranked[:1]
            riders = [track for track in objects if track not in heads]
            slot_of = self._slots.assign(heads)
            jobs = []
            for track in heads:
                own = (track, *riders) if track == heads[0] else (track,)
                rows = torch.tensor([objects.index(member) for member in own], device=object_ids.device)
                signature = (track,) + tuple(other for other in objects if other != track)
                previous_mask = None
                if not self.pool.holds_previous(slot_of[track], pair, signature):
                    if several and previous.reach is None:
                        previous = ObjectLayout.of(pair.previous_points, pair.previous_track_ids, object_ids, self.context_cell_m)
                    previous_mask = previous.cloud_mask(rows) if several else torch.isin(previous.row, rows)
                current_mask = current.cloud_mask(rows) if several else torch.isin(current.row, rows)
                jobs.append(ForwardJob(slot_of[track], own, signature, current_mask, previous_mask))

            results = self.pool.launch(pair, jobs, track_labels)
            self.pool.join(jobs)
            anchors = AnchorParts()
            for job in jobs:
                anchors.add_own(results[job.slot][0], object_ids[torch.tensor([objects.index(member) for member in job.own], device=object_ids.device)])
            self._ran = [job.slot for job in jobs]
            return anchors.flow_result(keep_mask_of(pair, jobs, results), pair.current_points.dtype)
