from __future__ import annotations

import numpy as np
import torch

from sam_rgbd_tracking.visualization import make_overlay

from .config import PipelineConfig
from .data_types import (
    MultiCameraFrame,
    SceneVelocityOutput,
    TrackedInstanceFrame,
)
from .flow_prediction import DifFlowPredictor
from .instance_filter import CommonInstanceFilter
from .profiler import CycleProfiler
from .tracker_adapter import MultiViewTrackerAdapter
from .velocity_recovery import VelocityRecovery


class ScenePredictionPipeline:
    """MultiViewRGBDTracker -> common-ID DifFlow3D -> dense velocity recovery."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        if config.runtime.allow_tf32 and torch.cuda.is_available():
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        # Load/patch the upstream tracker configuration once. DifFlow needs the
        # tracker output voxel resolution before any GPU model is constructed,
        # while its CUDA graphs still need to be captured before tracker CUDA
        # graphs/streams exist.
        tracker_config = MultiViewTrackerAdapter.prepare_native_config(config)
        upstream_voxel_size_m = MultiViewTrackerAdapter.output_voxel_size_m(
            tracker_config
        )

        self.flow_predictor = DifFlowPredictor(
            config,
            upstream_voxel_size_m=upstream_voxel_size_m,
        )
        self.flow_predictor.prepare()

        self.tracker = MultiViewTrackerAdapter(
            config,
            tracker_config=tracker_config,
        )
        self.instance_filter = CommonInstanceFilter()
        self.recovery = VelocityRecovery(config)
        self.profiler = CycleProfiler(config.runtime.enable_cuda_timing)
        self.previous_tracked: TrackedInstanceFrame | None = None
        self.last_flow_gap_s: float | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.runtime.device)

    def _empty_points(self) -> torch.Tensor:
        return torch.empty((0, 3), device=self.device, dtype=torch.float32)

    def _empty_ids(self) -> torch.Tensor:
        return torch.empty((0,), device=self.device, dtype=torch.int32)

    def _all_tracked_points(
        self,
        tracked: TrackedInstanceFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not tracked.instances:
            return self._empty_points(), self._empty_ids()
        points = torch.cat(
            [item.points_world for item in tracked.instances], dim=0
        ).contiguous()
        ids = torch.cat(
            [
                torch.full(
                    (item.points_world.shape[0],),
                    int(item.global_track_id),
                    device=points.device,
                    dtype=torch.int32,
                )
                for item in tracked.instances
            ],
            dim=0,
        )
        return points, ids

    def _visualization_payload(
        self,
        frame: MultiCameraFrame,
        tracked: TrackedInstanceFrame | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        annotated: dict[str, np.ndarray] = {}
        masks: dict[str, np.ndarray] = {}

        if tracked is None:
            if self.config.output.publish_annotated_rgb:
                annotated = {
                    name: np.ascontiguousarray(camera.rgb, dtype=np.uint8)
                    for name, camera in frame.cameras.items()
                }
            return masks, annotated

        for camera_name, result in tracked.view_results.items():
            if self.config.output.publish_annotated_rgb:
                annotated[camera_name] = make_overlay(result)
            if self.config.output.publish_tracked_masks:
                shape = result.frame.depth_m.shape
                combined = np.zeros(shape, dtype=bool)
                for instance in result.instances:
                    combined |= np.asarray(instance.mask, dtype=bool)
                masks[camera_name] = combined
        return masks, annotated

    def _empty_output(
        self,
        frame: MultiCameraFrame,
        tracked: TrackedInstanceFrame | None,
    ) -> SceneVelocityOutput:
        tracked_points = self._empty_points()
        tracked_ids = self._empty_ids()
        if tracked is not None:
            tracked_points, tracked_ids = self._all_tracked_points(tracked)
        masks, annotated = self._visualization_payload(frame, tracked)
        timings = self.profiler.finish()
        return SceneVelocityOutput(
            stamp_ns=int(frame.stamp_ns if tracked is None else tracked.stamp_ns),
            tracked_points=tracked_points,
            tracked_track_ids=tracked_ids,
            flow_points=self._empty_points(),
            flow_velocity=self._empty_points(),
            flow_track_ids=self._empty_ids(),
            source_anchors=self._empty_points(),
            warped_anchors=self._empty_points(),
            tracked_masks=masks,
            annotated_rgb=annotated,
            common_track_ids=(),
            flow_valid=False,
            timings_ms=timings,
        )

    def process(self, frame: MultiCameraFrame) -> SceneVelocityOutput:
        self.profiler.start_cycle()
        self.last_flow_gap_s = None

        with self.profiler.stage("tracker_total", cuda=False):
            current = self.tracker.process(frame)

        # The first synchronized bundle may be consumed solely by EfficientTAM
        # prewarm and is intentionally not a temporal source.
        if current is None:
            return self._empty_output(frame, None)

        for name, value in current.tracker_timings_ms.items():
            self.profiler.record(f"tracker/{name}", value)

        tracked_points, tracked_ids = self._all_tracked_points(current)

        previous = self.previous_tracked
        # Always advance to the immediately current tracker frame, even when the
        # pair is empty, stale, or skipped.
        self.previous_tracked = current

        with self.profiler.stage("instance_filter", cuda=True):
            pair = self.instance_filter.select(previous, current)

        flow_points = self._empty_points()
        flow_velocity = self._empty_points()
        flow_track_ids = self._empty_ids()
        source_anchors = self._empty_points()
        warped_anchors = self._empty_points()
        common_ids: tuple[int, ...] = ()
        flow_valid = False

        if pair is None:
            self.flow_predictor.reset()
        else:
            common_ids = pair.common_track_ids
            if pair.dt_s > float(self.config.flow.max_frame_gap_s):
                self.last_flow_gap_s = float(pair.dt_s)
                self.flow_predictor.reset()
            else:
                # DifFlow is always present. It owns adaptive voxel-2, exact-count
                # selection, frozen world/model scaling, and CUDA-Graph inference.
                with self.profiler.stage("difflow", cuda=True):
                    flow_result = self.flow_predictor.predict(pair)

                with self.profiler.stage("velocity_recovery", cuda=True):
                    flow_velocity = self.recovery.recover(pair, flow_result)

                flow_points = pair.current_points
                flow_track_ids = pair.current_track_ids
                source_anchors = flow_result.source_anchors
                warped_anchors = flow_result.warped_anchors
                flow_valid = True

        with self.profiler.stage("visualization_build", cuda=False):
            masks, annotated = self._visualization_payload(frame, current)

        timings = self.profiler.finish()
        return SceneVelocityOutput(
            stamp_ns=int(current.stamp_ns),
            tracked_points=tracked_points,
            tracked_track_ids=tracked_ids,
            flow_points=flow_points,
            flow_velocity=flow_velocity,
            flow_track_ids=flow_track_ids,
            source_anchors=source_anchors,
            warped_anchors=warped_anchors,
            tracked_masks=masks,
            annotated_rgb=annotated,
            common_track_ids=common_ids,
            flow_valid=flow_valid,
            timings_ms=timings,
        )

    def close(self) -> None:
        self.tracker.close()
