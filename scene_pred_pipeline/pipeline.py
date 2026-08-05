from __future__ import annotations

import torch

from .config import PipelineConfig
from .data_types import (
    MotionState,
    MultiCameraFrame,
    SceneVelocityOutput,
    TrackedFrame,
)
from .flow_prediction import DifFlowPredictor
from .multiview import MultiViewFusion
from .perception import PerViewPerception
from .profiler import CycleProfiler
from .sampling import MovingPointSampler
from .tracking import SimpleObjectTracker
from .velocity_recovery import VelocityRecovery
from .voxel import GlobalVoxelizer


class ScenePredictionPipeline:
    """Complete Step 1-5 scene prediction pipeline."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.perception = PerViewPerception(config)
        self.multiview = MultiViewFusion(config)
        self.voxelizer = GlobalVoxelizer(config)
        self.tracker = SimpleObjectTracker(config)
        self.sampler = MovingPointSampler(config)
        self.flow_predictor = DifFlowPredictor(config)
        self.recovery = VelocityRecovery(config)
        self.profiler = CycleProfiler(
            config.runtime.enable_cuda_timing
        )
        self.frame_index = 0
        self.previous_tracked: TrackedFrame | None = None
        self.last_flow_gap_s: float | None = None

    @staticmethod
    def _cat_or_empty(
        parts: list[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        return (
            torch.cat(parts, dim=0)
            if parts
            else torch.empty((0, 3), device=device)
        )

    @staticmethod
    def _moving_without_flow(
        tracked: TrackedFrame,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        moving_objects = [
            item
            for item in tracked.objects
            if item.motion_state == MotionState.MOVING
        ]
        if not moving_objects:
            return (
                torch.empty((0, 3), device=device),
                torch.empty((0, 3), device=device),
                torch.empty(
                    (0,),
                    device=device,
                    dtype=torch.int64,
                ),
            )

        points = torch.cat(
            [item.points for item in moving_objects],
            dim=0,
        )
        track_ids = torch.cat(
            [
                torch.full(
                    (item.points.shape[0],),
                    item.track_id,
                    device=device,
                    dtype=torch.int64,
                )
                for item in moving_objects
            ],
            dim=0,
        )
        return points, torch.zeros_like(points), track_ids

    def process(
        self,
        frame: MultiCameraFrame,
        profiler: "CycleProfiler | None" = None,
    ) -> dict[str, PerViewResult]:
        self.profiler.start_cycle()
        self.last_flow_gap_s = None

        # Step 1: YOLO, CLIP, masks and per-view point clouds.
        self.profiler.start("step1_perception")
        per_view = self.perception.process(
            frame,
            profiler=self.profiler,
        )
        self.profiler.stop("step1_perception")

        # Step 2: Multi-view object association and fusion.
        self.profiler.start("step2_multiview")
        fused = self.multiview.process(
            frame.stamp_ns,
            per_view,
        )
        self.profiler.stop("step2_multiview")

        # Step 3: Fixed world-frame voxel downsampling.
        self.profiler.start("step3_voxel")
        voxelized = self.voxelizer.process(
            self.frame_index,
            fused,
        )
        self.profiler.stop("step3_voxel")

        # Step 4: Tracking and moving/static classification.
        self.profiler.start("step4_tracking")
        tracked = self.tracker.update(voxelized)
        self.profiler.stop("step4_tracking")

        device = tracked.background_points.device

        static_points = self._cat_or_empty(
            [
                item.points
                for item in tracked.objects
                if item.motion_state == MotionState.STATIC
            ],
            device,
        )

        # Step 5A: Moving-point extraction, preselection and FPS.
        self.profiler.start("step5_sampling")
        flow_input = self.sampler.prepare(
            self.previous_tracked,
            tracked,
        )
        self.profiler.stop("step5_sampling")

        gap_is_valid = (
            flow_input is not None
            and flow_input.dt_s
            <= self.config.flow.max_frame_gap_s
        )

        if flow_input is None:
            (
                moving_points,
                moving_velocity,
                moving_track_ids,
            ) = self._moving_without_flow(
                tracked,
                device,
            )

        elif not gap_is_valid:
            # Skip this invalid temporal pair and make the current frame
            # the source frame for the next iteration.
            self.last_flow_gap_s = flow_input.dt_s
            self.flow_predictor.reset()

            moving_points = flow_input.current_candidates
            moving_velocity = torch.zeros_like(
                moving_points
            )
            moving_track_ids = (
                flow_input.current_candidate_track_ids
            )

        else:
            # Step 5B: DifFlow3D inference.
            self.profiler.start("step5_difflow")
            flow_result = self.flow_predictor.predict(
                flow_input,
                self.previous_tracked.stamp_ns,
                tracked.stamp_ns,
            )
            self.profiler.stop("step5_difflow")

            # Step 5C: Recover velocity to all moving candidate points.
            self.profiler.start("step5_recovery")
            moving_velocity = self.recovery.recover(
                flow_input,
                flow_result,
            )
            self.profiler.stop("step5_recovery")

            moving_points = flow_input.current_candidates
            moving_track_ids = (
                flow_input.current_candidate_track_ids
            )

        timings = self.profiler.finish()

        output = SceneVelocityOutput(
            stamp_ns=tracked.stamp_ns,
            background_points=tracked.background_points,
            static_points=static_points,
            moving_points=moving_points,
            moving_velocity=moving_velocity,
            moving_track_ids=moving_track_ids,
            moving_masks=tracked.moving_masks,
            timings_ms=timings,
        )

        # Always advance the temporal state, including after a large frame gap.
        self.previous_tracked = tracked
        self.frame_index += 1

        return output