"""object_flow.mode scheduled: the flow stage decides per cycle which objects get a forward and how they are packed.

Every object keeps a per-point network estimate with its own pyramid; joint and per_object are the two ends of what
this mode does between them.

  own forward   the deployed runner on its own CUDA stream, the neighbours' points as context, every cycle. Scarce
                (``own_forwards`` of them): they go to the objects that touch another object or are too large for a
                row, the ones that moved fastest recently first. What is left of those shares the last forward if it is
                too large for a row, and is a row otherwise.
  row           every other object - isolated ones, a table a person leans on, the second track of an object seen
                from two sides - is one row of a batched forward (flow_rows.py): small objects 256 anchors, larger
                ones 512, no context. A row is re-estimated every ``period`` cycles (short for fast objects and for
                objects whose velocity is changing) over the frames since its last estimate: a longer baseline for
                slow objects, less velocity noise. In between its anchors are carried with their own velocity and the
                object's measured acceleration, and the uncertainty grows with what is not known about the
                acceleration since.
  budget        a controller on the median time of the pipeline's cycle first postpones rows, then thins the own
                forwards' clouds; only when the cycle is clearly over the target does it give up the separate forwards
                (all objects with an own forward share one). With ten objects the flow stage is about a third of the
                cycle, so the controller's reach is a few milliseconds and it is bounded accordingly.

One cycle (``ScheduledFlowPredictor.predict``): plan -> launch the own forwards and the due rows, each on its own CUDA
stream -> wait -> store the new row estimates -> collect every object's anchors (own forwards, fresh and carried rows).

Measured (SafetyFilterBench, RTX 5090, fixed 30 Hz; whole cycle median, joint / per_object / scheduled): two objects
17.7 / 20.7 / 21.4 ms; person + stick + 8 balls 28.4 / 38.9 / 32.8 ms with the scheduled errors below joint in every
class; mixed shapes with furniture tracked as objects 31.5 / 44.7 / 35.6 ms.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair
from .flow_rows import RowBatch
from .object_flow import AnchorParts, ForwardJob, ForwardPool, ObjectLayout, SlotMap, _PoolBacked, keep_mask_of

SMALL_EXTENT_M = 0.35          # largest side of the bounding box up to which an object is a small row
OWN_EXTENT_M = 0.9             # an isolated object above this is a candidate for an own forward
ROW_EXTENT_MAX_M = 1.3         # a candidate above this is never a row
ROW_CLASSES = ((256, (3, 6, 12)), (512, (2, 4, 8)))    # (anchors per row, captured row counts): small, medium
MIN_OWN_POINTS = 256           # fewer points: a row
MIN_ROW_POINTS = 16            # fewer points: no estimate
PEAK_DECAY = 0.99              # per cycle, of an object's remembered peak speed (half-life about 2 s at 30 Hz)
ACCEL_LIMIT_MS2 = 5.0          # the measured acceleration a row is carried with is clipped to this
ACCEL_UNKNOWN_MS2 = 0.5        # floor of the acceleration a carried row's uncertainty allows for
SHARED = "shared"              # slot key of the forward that several objects share
GOLDEN = 0.6180339887          # thinning keeps the points whose index * GOLDEN has a fractional part below the density


# ----------------------------------------------------------------------------------------------------------------------
# Planning (host side, no tensors)
# ----------------------------------------------------------------------------------------------------------------------
def plan_objects(objects, count, extent, touching, peak_speed, own_forwards: int):
    """Who gets a forward of its own and who is a row. -> (own, [small rows, medium rows]).

    ``own`` is ordered: the first ``own_forwards`` (the fastest recently; an object not seen before counts as fastest)
    are served alone as far as the forwards go, see ``own_groups``.
    """
    candidates = sorted(
        (track for track in objects if count[track] >= MIN_OWN_POINTS and (touching[track] or extent[track] > OWN_EXTENT_M)),
        key=lambda track: (-peak_speed.get(track, float("inf")), -count[track]),
    )
    own = candidates[:own_forwards] + [track for track in candidates[own_forwards:] if extent[track] > ROW_EXTENT_MAX_M]
    rows: list[list[int]] = [[], []]
    for track in objects:
        if track not in own and count[track] >= MIN_ROW_POINTS:
            rows[0 if extent[track] <= SMALL_EXTENT_M else 1].append(track)
    return own, rows


def own_groups(own: list[int], forwards: int) -> list[tuple[int, ...]]:
    """The objects of each own forward: one each while the forwards last, the rest together in the last one."""
    alone = own[:forwards] if len(own) <= forwards else own[: forwards - 1]
    return [(track,) for track in alone] + ([tuple(own[len(alone):])] if len(own) > len(alone) else [])


def update_period(speed_ms: float, accel_ms2: float, multi_rate: bool) -> int:
    """Cycles until a row's next estimate. A slow object at a turning point has little speed and a large acceleration;
    carrying it goes wrong there first, so both decide."""
    if not multi_rate:
        return 1
    by_speed = 1 if speed_ms >= 1.0 else (2 if speed_ms >= 0.4 else 3)
    by_accel = 1 if accel_ms2 > 3.0 else (2 if accel_ms2 > 1.0 else 3)
    return min(by_speed, by_accel)


@dataclass(frozen=True)
class BudgetLevel:
    rows: int                  # small rows re-estimated per cycle at most (medium rows: half)
    density: float             # share of the own forwards' points that is used
    merge_own: bool            # all objects with an own forward share one


class BudgetController:
    """Follows the median of the last ``WINDOW`` cycles (a single slow cycle, e.g. while the segmentation model
    refreshes, is not something the flow stage can remove)."""

    LEVELS = (
        BudgetLevel(12, 1.0, False), BudgetLevel(6, 1.0, False), BudgetLevel(4, 1.0, False), BudgetLevel(3, 1.0, False),
        BudgetLevel(3, 0.85, False), BudgetLevel(2, 0.72, False),
        # overload: these cost accuracy and save little once the rest of the cycle dominates
        BudgetLevel(3, 1.0, True), BudgetLevel(2, 0.85, True), BudgetLevel(2, 0.72, True), BudgetLevel(1, 0.72, True),
    )
    NORMAL = 6                 # the levels after the first NORMAL are entered only OVERLOAD_MS above the target
    OVERLOAD_MS = 3.0
    RELAX_MS = 3.0             # below the target by this much: one level back
    WINDOW = 15

    def __init__(self, target_ms: float, enabled: bool) -> None:
        self.target_ms, self.enabled = float(target_ms), bool(enabled)
        self.index = 0
        self.median_ms: float | None = None
        self._history: list[float] = []
        self._hold = 0             # cycles until the next change

    @property
    def level(self) -> BudgetLevel:
        return self.LEVELS[self.index]

    def observe(self, cycle_ms: float) -> None:
        self._history = (self._history + [float(cycle_ms)])[-self.WINDOW:]
        self.median_ms = sorted(self._history)[len(self._history) // 2]
        if not self.enabled or len(self._history) < self.WINDOW:
            return
        self._hold = max(0, self._hold - 1)
        if self._hold:
            return
        over = self.median_ms - self.target_ms
        if (over > 0.0 and self.index < self.NORMAL - 1) or (over > self.OVERLOAD_MS and self.index < len(self.LEVELS) - 1):
            self.index += 1
            self._hold = 10
        elif over < -self.RELAX_MS and self.index > 0:
            self.index -= 1
            self._hold = 30


# ----------------------------------------------------------------------------------------------------------------------
# What is kept of a row between its estimates
# ----------------------------------------------------------------------------------------------------------------------
@dataclass
class RowTrack:
    anchors: torch.Tensor          # [n, 3] the object's anchors at its last estimate: the source of the next one
    stamp_ns: int                  # time of the last estimate
    cycle: int
    position: torch.Tensor         # [n, 3] where the last estimate put its source anchors at stamp_ns
    velocity: torch.Tensor         # [n, 3] m/s, the mean over the estimate's span
    cov: torch.Tensor | None       # [n, 3, 3] (m/s)^2
    mean_velocity: torch.Tensor    # [3]
    centre_s: float                # the time the mean velocity belongs to: the middle of the span
    span_s: float
    accel: torch.Tensor            # [3] m/s^2, from the last two estimates (clipped)
    accel_ms2: float = 0.0         # its norm
    period: int = 1                # cycles until the next estimate

    def carried(self, stamp_ns: int, dt: float):
        """The anchors at ``stamp_ns`` as a one-frame flow: (source, position, displacement over dt, the speed (m/s)
        the uncertainty has to grow by)."""
        age = (int(stamp_ns) - self.stamp_ns) * 1.0e-9
        lead = 0.5 * self.span_s + age                     # since the time the velocity belongs to
        velocity = self.velocity + (self.accel * lead)[None]
        position = self.position + age * velocity
        flow = velocity * dt
        return position - flow, position, flow, max(ACCEL_UNKNOWN_MS2, 0.5 * self.accel_ms2) * lead


def due_rows(objects, tracks: dict[int, RowTrack], cycle: int, limit: int, anchors: int) -> list[int]:
    """The objects whose row is re-estimated now: new ones first, then the most overdue, at most ``limit``."""
    due = []
    for track in objects:
        state = tracks.get(track)
        if state is None or state.anchors.shape[0] != anchors:
            due.append((float("inf"), track))
        elif cycle - state.cycle >= state.period:
            due.append(((cycle - state.cycle) / state.period, track))
    return [track for _, track in sorted(due, reverse=True)[:limit]]


@dataclass
class RowLaunch:
    """A batched forward under way."""

    batch: RowBatch
    objects: list[int]
    source: torch.Tensor           # [R, n, 3]
    target: torch.Tensor           # [R, n, 3]
    span_s: list[float]            # time from source to target, per row


# ----------------------------------------------------------------------------------------------------------------------
class ScheduledFlowPredictor(_PoolBacked):
    def __init__(
        self,
        config: PipelineConfig,
        *,
        upstream_voxel_size_m: float,
        predictor_factory: Callable[[], object] | None = None,
        row_batch_factory: Callable[[int, tuple[int, ...]], object] | None = None,
    ) -> None:
        object_flow = config.difflow.object_flow
        schedule = object_flow.scheduled
        self.config = config
        self.context_cell_m = float(object_flow.context_cell_m)
        self.multi_rate = bool(schedule.multi_rate)
        self.pool = ForwardPool(config, upstream_voxel_size_m=upstream_voxel_size_m, size=int(schedule.own_forwards), predictor_factory=predictor_factory)
        self.budget = BudgetController(schedule.target_cycle_ms, schedule.budget_control)
        self._row_batch_factory = row_batch_factory or self._deployed_row_batch
        self.row_batches: list = []                    # small, medium
        self._slots = SlotMap(len(self.pool))
        self._tracks: dict[int, RowTrack] = {}
        self._peak_speed: dict[int, float] = {}        # m/s, slowly decaying
        self._cycle = 0
        self._ran: list[int] = []
        self._schedule: dict[str, float] = {}          # what the last cycle did, reported with the stage times

    def _deployed_row_batch(self, anchors: int, row_counts: tuple[int, ...]) -> RowBatch:
        """Rows run the first runner's network and uncertainty head in the same canonical frame."""
        canonical = self.config.difflow.preprocessing.canonical_normalization
        first = self.pool.first
        return RowBatch(
            first.model, first.uncertainty_head, anchors=anchors, row_counts=row_counts,
            uncertainty=float(self.config.difflow.model.uncertainty), model_center=tuple(canonical.model_center),
            model_radius=float(canonical.model_radius), k_max=canonical.k_max, device=self.pool.device,
        )

    def prepare(self) -> None:
        self.pool.prepare()
        self.row_batches = [self._row_batch_factory(anchors, row_counts) for anchors, row_counts in ROW_CLASSES]
        for batch in self.row_batches:
            batch.prepare()

    def _forget(self) -> None:
        self._ran = []
        self._schedule = {}
        self._slots.clear()
        self._tracks.clear()
        self._peak_speed.clear()

    def observe_cycle_ms(self, cycle_ms: float) -> None:
        """The pipeline reports the time of its whole cycle after every frame."""
        self.budget.observe(cycle_ms)

    def resolve_after_cycle_fence(self):
        timing, info, debug = self.pool.resolve(self._ran)
        timing.update(self._schedule)
        return timing, info, debug

    # ---- one cycle ---------------------------------------------------------------------------------------------------
    def predict(self, pair: InstancePair, *, track_labels: dict[int, str] | None = None) -> FlowResult:
        objects = sorted(int(track) for track in pair.common_track_ids)
        if not objects:
            return self._joint(pair, track_labels)
        self._cycle += 1
        with torch.inference_mode():
            object_ids = torch.tensor(objects, device=pair.current_track_ids.device, dtype=pair.current_track_ids.dtype)
            current = ObjectLayout.of(pair.current_points, pair.current_track_ids, object_ids, self.context_cell_m)
            previous = ObjectLayout.of(pair.previous_points, pair.previous_track_ids, object_ids, self.context_cell_m, grid=False)
            own, rows = self._plan(objects, current, previous)
            level = self.budget.level

            jobs = self._own_jobs(pair, objects, object_ids, own, current, previous, level)
            results = self.pool.launch(pair, jobs, track_labels)
            launches = self._launch_rows(pair, objects, rows, current, previous, level)
            self.pool.join(jobs)
            for launch in launches:
                self._store_rows(launch, pair)

            anchors = AnchorParts()
            self._add_rows(anchors, [track for class_rows in rows for track in class_rows], pair)
            for job in jobs:
                anchors.add_own(results[job.slot][0], torch.tensor(job.own, device=object_ids.device, dtype=object_ids.dtype))
            self._remember_own_speeds(jobs, results, float(pair.dt_s))
            if not anchors:
                return self._joint(pair, track_labels)
            self._ran = [job.slot for job in jobs]
            self._schedule = {
                "schedule_own_objects": float(len(own)),
                "schedule_row_objects": float(sum(len(class_rows) for class_rows in rows)),
                "schedule_rows_estimated": float(sum(len(launch.objects) for launch in launches)),
                "schedule_budget_level": float(self.budget.index),
            }
            return anchors.flow_result(keep_mask_of(pair, jobs, results), pair.current_points.dtype)

    def _joint(self, pair: InstancePair, track_labels) -> FlowResult:
        """Nothing to schedule: the first runner on the pair as it is."""
        self._ran = [0]
        self._schedule = {}
        return self.pool.first.predict(pair, track_labels=track_labels)

    def _plan(self, objects: list[int], current: ObjectLayout, previous: ObjectLayout):
        count = torch.minimum(current.count, previous.count)
        host = torch.cat([count[:, None], current.extent[:, None], current.touching().float()], 1).tolist()   # the one host read
        number = len(objects)
        for track in [track for track in self._peak_speed if track not in objects]:
            del self._peak_speed[track]
        return plan_objects(
            objects,
            count={track: int(host[k][0]) for k, track in enumerate(objects)},
            extent={track: host[k][1] for k, track in enumerate(objects)},
            touching={track: any(host[k][2 + j] > 0 for j in range(number) if j != k) for k, track in enumerate(objects)},
            peak_speed=self._peak_speed,
            own_forwards=len(self.pool),
        )

    # ---- own forwards ------------------------------------------------------------------------------------------------
    def _own_jobs(self, pair, objects, object_ids, own, current, previous, level: BudgetLevel) -> list[ForwardJob]:
        groups = own_groups(own, 1 if level.merge_own else len(self.pool))
        keys = [group[0] if len(group) == 1 else SHARED for group in groups]
        slot_of = self._slots.assign(keys)
        thin = level.density < 1.0
        if thin and groups:      # the same low-discrepancy share of every cloud
            device = object_ids.device
            used_current = torch.frac(torch.arange(pair.current_points.shape[0], device=device) * GOLDEN) < level.density
            used_previous = torch.frac(torch.arange(pair.previous_points.shape[0], device=device) * GOLDEN) < level.density
        jobs = []
        for group, key in zip(groups, keys):
            slot = slot_of[key]
            rows = torch.tensor([objects.index(track) for track in group], device=object_ids.device)
            signature = group + tuple(track for track in objects if track not in group) + (level.density,)
            current_mask = current.cloud_mask(rows)
            previous_mask = None
            if not self.pool.holds_previous(slot, pair, signature):       # a new slot or a changed cloud: encode the source again
                if previous.reach is None:
                    previous = ObjectLayout.of(pair.previous_points, pair.previous_track_ids, object_ids, self.context_cell_m)
                previous_mask = previous.cloud_mask(rows)
            if thin:
                current_mask = current_mask & used_current
                previous_mask = None if previous_mask is None else previous_mask & used_previous
            jobs.append(ForwardJob(slot, group, signature, current_mask, previous_mask))
        return jobs

    def _remember_own_speeds(self, jobs, results, dt: float) -> None:
        """Peak speed (90th percentile over its anchors) of every object with an own forward."""
        measured = []
        for job in jobs:
            result = results[job.slot][0]
            for track in job.own:
                speed = result.anchor_flow[result.source_anchor_track_ids == track].float().norm(dim=1)
                if speed.numel() >= 8:
                    measured.append((track, torch.quantile(speed, 0.9)))
        if measured:
            for (track, _), speed in zip(measured, (torch.stack([value for _, value in measured]) / dt).tolist()):
                self._peak_speed[track] = max(speed, PEAK_DECAY * self._peak_speed.get(track, speed))

    # ---- rows --------------------------------------------------------------------------------------------------------
    def _launch_rows(self, pair, objects, rows, current: ObjectLayout, previous: ObjectLayout, level: BudgetLevel) -> list[RowLaunch]:
        """Start the batched forward of every row class that has rows due."""
        for track in [track for track in self._tracks if not any(track in class_rows for class_rows in rows)]:
            del self._tracks[track]
        now = int(pair.current_stamp_ns)
        launches = []
        for index, (batch, class_rows) in enumerate(zip(self.row_batches, rows)):
            limit = min(level.rows if index == 0 else max(1, level.rows // 2), batch.capacity)
            run = due_rows(class_rows, self._tracks, self._cycle, limit, batch.anchors)
            if not run:
                continue
            # A row runs from its last estimate to now; a new row (or one that changed its class) over this pair.
            known = [track in self._tracks and self._tracks[track].anchors.shape[0] == batch.anchors for track in run]
            spans = [(now - self._tracks[track].stamp_ns) * 1.0e-9 if old else float(pair.dt_s) for track, old in zip(run, known)]
            target = current.evenly_spaced([objects.index(track) for track in run], batch.anchors)
            if all(known):
                source = torch.stack([self._tracks[track].anchors for track in run])
            else:
                source = previous.evenly_spaced([objects.index(track) for track in run], batch.anchors)
                for j, track in enumerate(run):
                    if known[j]:
                        source[j] = self._tracks[track].anchors
            batch.launch(source, target)
            launches.append(RowLaunch(batch, run, source, target, spans))
        return launches

    def _store_rows(self, launch: RowLaunch, pair: InstancePair) -> None:
        """A finished batched forward becomes the rows' new estimates, periods and peak speeds."""
        now = int(pair.current_stamp_ns)
        displacement, covariance = launch.batch.collect()
        span = torch.tensor([max(value, 1.0e-3) for value in launch.span_s], device=displacement.device)
        velocity = displacement / span[:, None, None]
        mean_velocity = velocity.mean(1)
        for j, track in enumerate(launch.objects):
            old = self._tracks.get(track)
            centre = now * 1.0e-9 - 0.5 * launch.span_s[j]
            accel = torch.zeros(3, device=velocity.device)
            if old is not None and centre - old.centre_s > 1.0e-3:
                accel = (mean_velocity[j] - old.mean_velocity) / (centre - old.centre_s)
                accel = accel * torch.clamp(ACCEL_LIMIT_MS2 / accel.norm().clamp_min(1.0e-9), max=1.0)
            self._tracks[track] = RowTrack(
                anchors=launch.target[j].clone(), stamp_ns=now, cycle=self._cycle,
                position=(launch.source[j] + displacement[j]).clone(), velocity=velocity[j].clone(),
                cov=None if covariance is None else (covariance[j] / span[j] ** 2).clone(),
                mean_velocity=mean_velocity[j].clone(), centre_s=centre, span_s=launch.span_s[j], accel=accel,
            )
        speed = velocity.norm(dim=-1).quantile(0.9, dim=1)
        accel = torch.stack([self._tracks[track].accel for track in launch.objects]).norm(dim=1)
        for track, (speed_ms, accel_ms2) in zip(launch.objects, torch.stack([speed, accel], 1).tolist()):
            state = self._tracks[track]
            state.accel_ms2, state.period = accel_ms2, update_period(speed_ms, accel_ms2, self.multi_rate)
            self._peak_speed[track] = max(speed_ms, (PEAK_DECAY ** state.period) * self._peak_speed.get(track, speed_ms))

    def _add_rows(self, anchors: AnchorParts, row_objects, pair: InstancePair) -> None:
        """Every row object's anchors: fresh, or carried since its last estimate."""
        dt, now = float(pair.dt_s), int(pair.current_stamp_ns)
        calibration_scale = self.calibration_scale
        for track in row_objects:
            state = self._tracks.get(track)
            if state is None:
                continue
            source, position, flow, growth_ms = state.carried(now, dt)
            ids = torch.full((position.shape[0],), track, device=position.device, dtype=pair.current_track_ids.dtype)
            if state.cov is None or calibration_scale is None:
                anchors.add(source, position, flow, ids)
                continue
            # The calibrated region is scale * sqrt(cov): the growth is added in the covariance's own units.
            scale = calibration_scale(flow)
            grown = state.cov * dt ** 2 + ((growth_ms * dt / scale) ** 2)[:, None, None] * torch.eye(3, device=position.device)[None]
            anchors.add(source, position, flow, ids, grown, scale)
