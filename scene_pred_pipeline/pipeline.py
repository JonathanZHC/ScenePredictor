from __future__ import annotations

from collections import defaultdict

import cv2
import numpy as np
import torch

from .config import PipelineConfig
from .data_types import (
    ImageDetection,
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
    """YOLOE + spatial association + DifFlow3D safety-filter pipeline."""

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config
        torch.backends.cuda.matmul.allow_tf32 = (
            config.runtime.allow_tf32
        )
        torch.backends.cudnn.allow_tf32 = (
            config.runtime.allow_tf32
        )

        # YOLOE performs its configured dummy warmup here, before any profiler
        # sample or ROS frame is processed.
        self.perception = PerViewPerception(config)
        self.multiview = MultiViewFusion(config)
        self.voxelizer = GlobalVoxelizer(config)
        self.tracker = SimpleObjectTracker(config)
        self.sampler = MovingPointSampler(config)
        self.flow_predictor = DifFlowPredictor(config)
        self.recovery = VelocityRecovery(config)

        self.profiler = CycleProfiler(
            enabled=config.runtime.enable_profiling,
            use_cuda_events=(
                config.runtime.enable_cuda_timing
            ),
            history_size=(
                config.runtime.profile_history_size
            ),
        )

        self.frame_index = 0
        self.previous_tracked: TrackedFrame | None = None
        self.last_flow_gap_s: float | None = None

    @staticmethod
    def _cat_or_empty(
        parts: list[torch.Tensor],
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        return (
            torch.cat(parts, dim=0)
            if parts
            else torch.empty(
                (0, 3),
                device=device,
                dtype=dtype,
            )
        )

    @staticmethod
    def _moving_without_flow(
        tracked: TrackedFrame,
        device: torch.device,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        moving_objects = [
            item
            for item in tracked.objects
            if item.motion_state == MotionState.MOVING
        ]
        if not moving_objects:
            return (
                torch.empty(
                    (0, 3),
                    device=device,
                ),
                torch.empty(
                    (0, 3),
                    device=device,
                ),
                torch.empty(
                    (0,),
                    device=device,
                    dtype=torch.int64,
                ),
            )

        points = torch.cat(
            [
                item.points
                for item in moving_objects
            ],
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
        return (
            points,
            torch.zeros_like(points),
            track_ids,
        )

    @staticmethod
    def _detections(
        tracked: TrackedFrame,
    ) -> dict[str, list[ImageDetection]]:
        output: dict[
            str,
            list[ImageDetection],
        ] = defaultdict(list)
        for item in tracked.objects:
            for member in item.members:
                output[
                    member.camera_name
                ].append(
                    ImageDetection(
                        bbox_xyxy=member.bbox_xyxy,
                        class_id=member.class_id,
                        class_name=member.class_name,
                        confidence=(
                            member.class_confidence
                        ),
                        track_id=item.track_id,
                        motion_state=item.motion_state,
                    )
                )
        return output

    @staticmethod
    def _annotate_rgb(
        frame: MultiCameraFrame,
        detections: dict[
            str,
            list[ImageDetection],
        ],
    ) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        for camera_name, camera in frame.cameras.items():
            image = np.ascontiguousarray(
                camera.rgb.copy(),
                dtype=np.uint8,
            )
            height, width = image.shape[:2]

            for detection in detections.get(
                camera_name,
                [],
            ):
                x0, y0, x1, y1 = (
                    detection.bbox_xyxy
                )
                x0 = max(
                    0,
                    min(width - 1, x0),
                )
                x1 = max(
                    0,
                    min(width - 1, x1),
                )
                y0 = max(
                    0,
                    min(height - 1, y0),
                )
                y1 = max(
                    0,
                    min(height - 1, y1),
                )

                moving = (
                    detection.motion_state
                    == MotionState.MOVING
                )
                # The image is RGB. OpenCV writes the tuple directly into the
                # channels, so these are RGB values rather than BGR semantics.
                color = (
                    (240, 50, 30)
                    if moving
                    else (40, 220, 80)
                )
                state = "M" if moving else "S"
                label = (
                    f"{detection.class_name} "
                    f"{detection.confidence:.2f} "
                    f"id={detection.track_id} "
                    f"{state}"
                )

                cv2.rectangle(
                    image,
                    (x0, y0),
                    (x1, y1),
                    color,
                    2,
                    lineType=cv2.LINE_AA,
                )
                (
                    text_width,
                    text_height,
                ), baseline = cv2.getTextSize(
                    label,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    1,
                )
                text_top = max(
                    0,
                    y0 - text_height - baseline - 4,
                )
                cv2.rectangle(
                    image,
                    (x0, text_top),
                    (
                        min(
                            width - 1,
                            x0 + text_width + 6,
                        ),
                        y0,
                    ),
                    color,
                    thickness=-1,
                )
                cv2.putText(
                    image,
                    label,
                    (
                        x0 + 3,
                        max(
                            text_height + 1,
                            y0 - baseline - 2,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    lineType=cv2.LINE_AA,
                )

            output[camera_name] = image
        return output

    def process(
        self,
        frame: MultiCameraFrame,
    ) -> SceneVelocityOutput:
        self.profiler.start_cycle()
        self.last_flow_gap_s = None

        self.profiler.start("step1_perception")
        per_view = self.perception.process(
            frame,
            profiler=self.profiler,
        )
        self.profiler.stop("step1_perception")

        self.profiler.start("step2_multiview")
        fused = self.multiview.process(
            frame.stamp_ns,
            per_view,
        )
        self.profiler.stop("step2_multiview")

        self.profiler.start("step3_voxel")
        voxelized = self.voxelizer.process(
            self.frame_index,
            fused,
        )
        self.profiler.stop("step3_voxel")

        self.profiler.start("step4_tracking")
        tracked = self.tracker.update(
            voxelized
        )
        self.profiler.stop("step4_tracking")

        device = tracked.background_points.device
        static_points = self._cat_or_empty(
            [
                item.points
                for item in tracked.objects
                if (
                    item.motion_state
                    == MotionState.STATIC
                )
            ],
            device,
        )

        if self.config.flow.enabled:
            # Exactly one FPS is performed for this current frame and the
            # result is retained in tracked.flow_cache.
            self.profiler.start(
                "step5_current_fps_cache"
            )
            self.sampler.cache_current(
                tracked
            )
            self.profiler.stop(
                "step5_current_fps_cache"
            )

            self.profiler.start(
                "step5_pair_from_cache"
            )
            flow_input = self.sampler.prepare(
                self.previous_tracked,
                tracked,
            )
            self.profiler.stop(
                "step5_pair_from_cache"
            )
        else:
            flow_input = None

        gap_is_valid = (
            flow_input is not None
            and flow_input.dt_s
            <= self.config.flow.max_frame_gap_s
        )

        if (
            not self.config.flow.enabled
            or flow_input is None
        ):
            (
                moving_points,
                moving_velocity,
                moving_track_ids,
            ) = self._moving_without_flow(
                tracked,
                device,
            )
        elif not gap_is_valid:
            self.last_flow_gap_s = (
                flow_input.dt_s
            )
            self.flow_predictor.reset()
            moving_points = (
                flow_input.current_candidates
            )
            moving_velocity = torch.zeros_like(
                moving_points
            )
            moving_track_ids = (
                flow_input.current_candidate_track_ids
            )
        else:
            self.profiler.start("step5_difflow")
            flow_result = (
                self.flow_predictor.predict(
                    flow_input,
                    self.previous_tracked.stamp_ns,
                    tracked.stamp_ns,
                )
            )
            self.profiler.stop("step5_difflow")

            self.profiler.start("step5_recovery")
            moving_velocity = (
                self.recovery.recover(
                    flow_input,
                    flow_result,
                )
            )
            self.profiler.stop("step5_recovery")

            moving_points = (
                flow_input.current_candidates
            )
            moving_track_ids = (
                flow_input.current_candidate_track_ids
            )

        if (
            self.config.runtime.enable_visualization
            and self.config.output.publish_annotated_rgb
        ):
            image_detections = self._detections(
                tracked
            )
            annotated_rgb = self._annotate_rgb(
                frame,
                image_detections,
            )
        else:
            annotated_rgb = {}

        timings = self.profiler.finish()
        output = SceneVelocityOutput(
            stamp_ns=tracked.stamp_ns,
            background_points=(
                tracked.background_points
            ),
            static_points=static_points,
            moving_points=moving_points,
            moving_velocity=moving_velocity,
            moving_track_ids=moving_track_ids,
            moving_masks=tracked.moving_masks,
            # Reuse original ROS NumPy arrays; no GPU RGB copy-back.
            camera_rgb={
                camera_name: camera.rgb
                for camera_name, camera
                in frame.cameras.items()
            },
            annotated_rgb=annotated_rgb,
            timings_ms=timings,
        )

        self.previous_tracked = tracked
        self.frame_index += 1
        return output
