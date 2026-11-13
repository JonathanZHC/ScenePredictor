"""Per-object scene flow: one DifFlow3D streaming runner per tracked object, run side by side.

Why: in one joint cloud the pyramid's coarse levels are sampled by spatial extent, so a thin or small object next
to a large one keeps only a handful of coarse points (a 0.6 m stick beside a person: 21 of 256, 6 of 64). The
coarse flow estimate at its extremity is then systematically wrong, the fine levels recover it only partly, and
the uncertainty head (correctly) reports a large region there. Inferred on its own, every object has a full
pyramid.

How:
  * every object slot owns a ``DifFlowPredictor`` (its CUDA graphs, point buckets, canonical normalisation and
    encoding reuse; model, iterations and shortcuts exactly as configured);
  * an object's cloud is ALL its points plus, as context, the other objects' points in the grid cells (edge
    ``context_cell_m``) that the object occupies or that touch one it occupies. Objects in contact need each
    other (the hand that holds a stick anchors the stick's motion and the other way round): without context the
    fast-motion error grows, with too much the small object is crowded out again. The cells are found with one
    scatter, one 3x3x3 max-pool and one gather per object - no distance matrix. The context depends on the frame
    alone, so a frame's cloud is the same as the target of one pair and the source of the next and the runner's
    encoding reuse stays valid;
  * the forwards run on separate CUDA streams and overlap on the GPU (RTX 5090: about 3.6 ms + 0.67 ms per 1000
    points + 0.9 ms per further object, instead of the sum of the single forwards);
  * only an object's OWN anchors are returned. The same points estimated as context in a neighbour's forward are
    worse and are dropped, so dense recovery stays same-track as before;
  * objects beyond the available slots and objects with fewer than ``min_points`` points ride in the largest
    object's cloud (joint with it, their anchors kept).

With a single tracked object the result is the plain joint forward.
"""
from __future__ import annotations

import contextlib
from typing import Callable

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair
from .flow_prediction import DifFlowPredictor


def context_cells(points: torch.Tensor, cell_m: float) -> tuple[torch.Tensor, list[int]]:
    """Integer cell of every point ``[N, 3]`` on a grid of edge ``cell_m`` and the grid size (one host read)."""
    cells = torch.floor((points - (points.min(0).values - cell_m)) / cell_m).long()
    return cells, (cells.max(0).values + 2).tolist()


def near_object(cells: tuple[torch.Tensor, list[int]], own: torch.Tensor) -> torch.Tensor:
    """Mask ``[N]`` of the points in a cell that the ``own`` points occupy or that touches one (own points included)."""
    index, dims = cells
    grid = torch.zeros(dims, device=index.device)
    grid[index[own, 0], index[own, 1], index[own, 2]] = 1.0
    grid = torch.nn.functional.max_pool3d(grid[None, None], 3, 1, 1)[0, 0]
    return grid[index[:, 0], index[:, 1], index[:, 2]] > 0


class PerObjectFlowPredictor:
    """Drop-in for ``DifFlowPredictor`` in ``ScenePredictorPipeline`` (same calls, same ``FlowResult``)."""

    def __init__(
        self,
        config: PipelineConfig,
        *,
        upstream_voxel_size_m: float,
        max_objects: int,
        predictor_factory: Callable[[], object] | None = None,
    ) -> None:
        if max_objects < 1:
            raise ValueError("max_objects must be >= 1")
        self.config = config
        self.device = torch.device(config.runtime.device)
        per_object = config.difflow.per_object
        self.context_cell_m = float(per_object.context_cell_m)
        self.min_points = int(per_object.min_points)
        factory = predictor_factory or (
            lambda: DifFlowPredictor(config, upstream_voxel_size_m=upstream_voxel_size_m)
        )
        self.predictors = [factory() for _ in range(int(max_objects))]
        self._streams: list[torch.cuda.Stream] | None = None
        self._slot_of: dict[int, int] = {}
        self._ran: list[int] = []

    # ---- the DifFlowPredictor surface the pipeline uses -------------------------------------------------
    @property
    def calibration_scale(self):
        return self.predictors[0].calibration_scale

    @property
    def outlier_filter_enabled(self) -> bool:
        return self.predictors[0].outlier_filter_enabled

    @property
    def detailed_outlier_output(self) -> bool:
        return self.predictors[0].detailed_outlier_output

    def prepare(self) -> None:
        """Capture every slot's CUDA graphs (before the tracker's graphs exist) and create the streams."""
        for predictor in self.predictors:
            predictor.prepare()
        if self.device.type == "cuda":
            # same (highest) priority as the pipeline's shared stream (cuda_streams.py): the asynchronous SAM3
            # refresh must not delay a forward
            _, greatest = torch.cuda.Stream.priority_range()
            self._streams = [torch.cuda.Stream(device=self.device, priority=int(greatest)) for _ in self.predictors]

    def reset(self) -> None:
        for predictor in self.predictors:
            predictor.reset()
        self._slot_of.clear()
        self._ran = []

    def reset_calibration(self) -> None:
        for predictor in self.predictors:
            predictor.reset_calibration()
        self._slot_of.clear()
        self._ran = []

    def resolve_after_cycle_fence(self) -> tuple[dict[str, float], dict[str, object], dict[str, object]]:
        """Timing and outlier statistics of the slots that ran this cycle. The forwards overlap, so a stage's time
        is the largest over the objects; voxel counts add up."""
        timing: dict[str, float] = {}
        info: dict[str, object] = {}
        debug: dict[str, object] = {}
        for slot in self._ran:
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
        if self._ran:
            timing["runner_objects"] = float(len(self._ran))
        return timing, info, debug

    # ---- per-object inference ----------------------------------------------------------------------------
    def _stream(self, slot: int):
        return contextlib.nullcontext() if self._streams is None else torch.cuda.stream(self._streams[slot])

    def _assign(self, pair: InstancePair) -> tuple[list[int], list[int]]:
        """(objects with a forward of their own, objects riding with the largest); keeps the slot of every object
        that stays (a slot's runner holds that object's previous frame)."""
        objects = sorted(int(track) for track in pair.common_track_ids)
        count = {
            track: min(int((pair.previous_track_ids == track).sum()), int((pair.current_track_ids == track).sum()))
            for track in objects
        }
        ranked = sorted(objects, key=lambda track: -count[track])
        heads = [track for track in ranked[: len(self.predictors)] if count[track] >= self.min_points] or ranked[:1]
        riders = [track for track in objects if track not in heads]
        for track in [track for track in self._slot_of if track not in heads]:
            del self._slot_of[track]
        for track in heads:
            if track not in self._slot_of:
                used = set(self._slot_of.values())
                self._slot_of[track] = next(slot for slot in range(len(self.predictors)) if slot not in used)
        return heads, riders

    def predict(self, pair: InstancePair, *, track_labels: dict[int, str] | None = None) -> FlowResult:
        objects = sorted(int(track) for track in pair.common_track_ids)
        if not objects:
            self._ran = [0]
            return self.predictors[0].predict(pair, track_labels=track_labels)
        with torch.inference_mode():
            heads, riders = self._assign(pair)
            previous_points, previous_ids = pair.previous_points, pair.previous_track_ids
            current_points, current_ids = pair.current_points, pair.current_track_ids
            several = len(objects) > 1
            current_cells = context_cells(current_points, self.context_cell_m) if several else None
            previous_cells = None
            jobs = []
            for track in heads:
                mine = torch.tensor(
                    [track] + (riders if track == heads[0] else []), device=current_ids.device, dtype=current_ids.dtype
                )
                slot = self._slot_of[track]
                signature = (track,) + tuple(other for other in objects if other != track)
                predictor = self.predictors[slot]
                own_current = torch.isin(current_ids, mine)
                current_mask = (own_current | near_object(current_cells, own_current)) if several else own_current
                cached = (
                    getattr(predictor, "_cached_target_stamp_ns", None) == int(pair.previous_stamp_ns)
                    and getattr(predictor, "_cached_track_signature", None) == signature
                )
                if cached:
                    previous_mask = None        # the runner still holds this frame's encoding: the source cloud is not read
                else:
                    own_previous = torch.isin(previous_ids, mine)
                    if several:
                        if previous_cells is None:
                            previous_cells = context_cells(previous_points, self.context_cell_m)
                        previous_mask = own_previous | near_object(previous_cells, own_previous)
                    else:
                        previous_mask = own_previous
                jobs.append((slot, mine, previous_mask, current_mask, signature))

            if self._streams is not None:
                # the inputs are complete before another stream reads them
                torch.cuda.current_stream(self.device).synchronize()
            results = {}
            for slot, mine, previous_mask, current_mask, signature in jobs:
                with self._stream(slot):
                    sub = InstancePair(
                        previous_stamp_ns=pair.previous_stamp_ns,
                        current_stamp_ns=pair.current_stamp_ns,
                        common_track_ids=signature,
                        previous_points=previous_points if previous_mask is None else previous_points[previous_mask],
                        current_points=current_points[current_mask],
                        previous_track_ids=previous_ids if previous_mask is None else previous_ids[previous_mask],
                        current_track_ids=current_ids[current_mask],
                        dt_s=pair.dt_s,
                    )
                    results[slot] = (self.predictors[slot].predict(sub, track_labels=track_labels), sub)
            if self._streams is not None:
                for slot, *_ in jobs:
                    self._streams[slot].synchronize()        # join: the forwards ran side by side

            keep_current = torch.ones(current_points.shape[0], dtype=torch.bool, device=current_points.device)
            source, warped, flow, ids, cov, scale = [], [], [], [], [], []
            for slot, mine, previous_mask, current_mask, signature in jobs:
                result, sub = results[slot]
                own = torch.isin(result.source_anchor_track_ids, mine)
                source.append(result.source_anchors[own])
                warped.append(result.warped_anchors[own])
                flow.append(result.anchor_flow[own])
                ids.append(result.source_anchor_track_ids[own])
                if result.anchor_flow_cov is not None:
                    cov.append(result.anchor_flow_cov[own])
                    scale.append(result.anchor_flow_cov_scale[own])
                own_current = torch.isin(sub.current_track_ids, mine)
                keep_current[current_mask.nonzero(as_tuple=True)[0][own_current]] = result.target_input_keep_mask[own_current]
            self._ran = [slot for slot, *_ in jobs]
            return FlowResult(
                source_anchors=torch.cat(source),
                warped_anchors=torch.cat(warped),
                anchor_flow=torch.cat(flow),
                source_anchor_track_ids=torch.cat(ids),
                target_input_keep_mask=keep_current,
                anchor_flow_cov=torch.cat(cov) if cov else None,
                anchor_flow_cov_scale=torch.cat(scale) if scale else None,
            )
