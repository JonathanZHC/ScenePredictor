from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable
import time

import numpy as np
import torch

from sam_rgbd_tracking.async_sam3 import AsyncSAM3Worker
from sam_rgbd_tracking.config import Config as TrackerNativeConfig
from sam_rgbd_tracking.config import load_config as load_tracker_config
from sam_rgbd_tracking.multiview_component import MultiViewEfficientTAMComponent

from .config import PipelineConfig
from .data_types import MultiCameraFrame, TrackedInstance, TrackedInstanceFrame


class MultiViewTrackerAdapter:
    """Direct in-process bridge to MultiViewRGBDTracker.

    EfficientTAM state is always constructed and touched by one owner thread.
    SAM3 retains the dependency's separate asynchronous worker/CUDA stream.
    No tracker data is serialized through ROS between tracking and scene flow.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        tracker_config: TrackerNativeConfig | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.camera_names = [str(name) for name in config.ros.camera_names]

        if tracker_config is None:
            tracker_config = self.prepare_native_config(config)
        self.tracker_config = tracker_config

        self._owner = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="scene-predictor-tracker-owner",
        )
        self.component: MultiViewEfficientTAMComponent = self._run_owner(
            MultiViewEfficientTAMComponent,
            tracker_config,
            camera_names=self.camera_names,
        )
        self.sam3_worker = AsyncSAM3Worker(tracker_config)

        self._prewarm_pending = bool(
            tracker_config.tracker.efficient_tam.get("prewarm_enabled", True)
        )
        self._initialized = False

    def _run_owner(self, function: Callable[..., Any], /, *args, **kwargs):
        return self._owner.submit(function, *args, **kwargs).result()

    @staticmethod
    def _resolve_checkpoint(value: str, checkpoint_root: Path) -> str:
        path = Path(str(value)).expanduser()
        if path.is_absolute():
            return str(path)
        parts = path.parts
        if parts and parts[0] == "checkpoints":
            path = Path(*parts[1:])
        return str((checkpoint_root / path).resolve())

    @classmethod
    def prepare_native_config(cls, config: PipelineConfig) -> TrackerNativeConfig:
        """Load and patch the upstream tracker config without constructing GPU models."""
        native = load_tracker_config(config.tracker.config_path)
        data = native.as_dict()
        camera_names = [str(name) for name in config.ros.camera_names]

        data.setdefault("runtime", {})["camera_names"] = camera_names

        # SAM3's image-model builder currently moves the model to CUDA only for
        # the literal device string "cuda". ScenePredictor commonly uses
        # "cuda:0", which would otherwise leave SAM3 weights on CPU while the
        # processor creates CUDA inputs. Normalize GPU 0 to the dependency's
        # native "cuda" spelling; keep CPU unchanged.
        requested_device = torch.device(config.runtime.device)
        if requested_device.type == "cuda":
            if requested_device.index not in (None, 0):
                raise ValueError(
                    "MultiViewRGBDTracker/SAM3 currently expects GPU 0 via "
                    "runtime.device='cuda'. Requested "
                    f"{config.runtime.device!r}."
                )
            data["runtime"]["device"] = "cuda"
        else:
            data["runtime"]["device"] = str(requested_device)

        data["runtime"]["enable_tf32"] = bool(config.runtime.allow_tf32)

        checkpoint_root = Path(config.tracker.checkpoint_root).expanduser()
        detector = data.setdefault("detector", {})
        if "checkpoint" in detector:
            detector["checkpoint"] = cls._resolve_checkpoint(
                str(detector["checkpoint"]), checkpoint_root
            )

        efficient = data.setdefault("tracker", {}).setdefault("efficient_tam", {})
        if "checkpoint" in efficient:
            efficient["checkpoint"] = cls._resolve_checkpoint(
                str(efficient["checkpoint"]), checkpoint_root
            )

        return TrackerNativeConfig(data)

    @staticmethod
    def output_voxel_size_m(tracker_config: TrackerNativeConfig) -> float:
        """Return the world-voxel resolution actually emitted by the tracker.

        DifFlow uses this only as the base resolution for its adaptive second
        voxel stage. Keeping the value owned by MultiViewRGBDTracker prevents
        ScenePredictor and tracking.yaml from drifting out of sync.
        """
        try:
            voxel_size_m = float(
                tracker_config.shared_voxel_grid.voxel_size_m
            )
        except (AttributeError, TypeError) as exc:
            raise ValueError(
                "MultiViewRGBDTracker config must define "
                "shared_voxel_grid.voxel_size_m"
            ) from exc
        if voxel_size_m <= 0.0:
            raise ValueError(
                "shared_voxel_grid.voxel_size_m must be positive, got "
                f"{voxel_size_m}"
            )
        return voxel_size_m

    def _view_inputs(self, frame: MultiCameraFrame) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
        view_inputs: list[dict[str, Any]] = []
        rgbs: list[np.ndarray] = []
        for camera_name in self.camera_names:
            camera = frame.cameras[camera_name]
            K = camera.K
            rgb = np.ascontiguousarray(camera.rgb, dtype=np.uint8)
            depth = np.ascontiguousarray(camera.depth, dtype=np.float32)
            view_inputs.append(
                {
                    "rgb": rgb,
                    "depth_m": depth,
                    "fx": float(K[0, 0]),
                    "fy": float(K[1, 1]),
                    "cx": float(K[0, 2]),
                    "cy": float(K[1, 2]),
                    "timestamp_ns": int(camera.stamp_ns),
                    "world_from_camera": np.asarray(
                        camera.T_world_camera, dtype=np.float32
                    ),
                }
            )
            rgbs.append(rgb)
        return view_inputs, rgbs

    def _to_tracked_frame(self, results: list[Any]) -> TrackedInstanceFrame:
        adapter_started = time.perf_counter()
        groups_started = time.perf_counter()
        groups = self._run_owner(self.component.get_last_multiview_instances)
        adapter_get_groups_ms = 1000.0 * (time.perf_counter() - groups_started)

        valid_groups = sorted(
            (
                group
                for group in groups
                if group.global_track_id is not None
                and group.points_world is not None
                and len(group.points_world) > 0
            ),
            key=lambda group: int(group.global_track_id),
        )

        instances: list[TrackedInstance] = []
        track_offsets: dict[int, tuple[int, int]] = {}
        track_ids = tuple(int(group.global_track_id) for group in valid_groups)
        adapter_prepare_started = time.perf_counter()

        if valid_groups:
            lengths = [int(len(group.points_world)) for group in valid_groups]

            # CrossFrameAligner already uploads every fused cloud for Chamfer.
            # Reuse those exact CUDA-bank views; silently falling back to another
            # CPU->GPU upload would hide a performance regression.
            ready_event = valid_groups[0].points_world_gpu_ready_event
            if ready_event is not None:
                torch.cuda.current_stream(self.device).wait_event(ready_event)

            clouds_gpu: list[torch.Tensor] = []
            offset = 0
            for group, count in zip(valid_groups, lengths):
                cloud_gpu = group.points_world_gpu
                if not isinstance(cloud_gpu, torch.Tensor) or not cloud_gpu.is_cuda:
                    raise RuntimeError(
                        "ScenePredictor requires CrossFrame GPU-bank cloud views; "
                        "the tracker returned a CPU-only fused cloud."
                    )
                if cloud_gpu.device != self.device:
                    raise RuntimeError(
                        f"Tracker cloud is on {cloud_gpu.device}, expected {self.device}."
                    )
                if cloud_gpu.dtype != torch.float32:
                    raise RuntimeError(
                        f"Tracker cloud dtype is {cloud_gpu.dtype}, expected float32."
                    )
                if int(cloud_gpu.shape[0]) != count:
                    raise RuntimeError(
                        "Tracker CPU/GPU fused-cloud sizes disagree: "
                        f"{count} != {int(cloud_gpu.shape[0])}."
                    )

                track_id = int(group.global_track_id)
                end_offset = offset + count
                track_offsets[track_id] = (offset, end_offset)
                clouds_gpu.append(cloud_gpu)
                offset = end_offset

            adapter_prepare_ms = 1000.0 * (
                time.perf_counter() - adapter_prepare_started
            )
            adapter_pack_started = time.perf_counter()

            # The CrossFrame bank is ping-pong workspace and will be overwritten.
            # One D2D cat creates frame-owned storage without another CPU round trip.
            packed_gpu = torch.cat(clouds_gpu, dim=0)
            id_values = torch.tensor(track_ids, dtype=torch.int32, device=self.device)
            repeats = torch.tensor(lengths, dtype=torch.long, device=self.device)
            packed_ids_gpu = torch.repeat_interleave(
                id_values, repeats, output_size=int(sum(lengths))
            )
            adapter_pack_submit_ms = 1000.0 * (
                time.perf_counter() - adapter_pack_started
            )

            for group in valid_groups:
                track_id = int(group.global_track_id)
                begin, end_offset = track_offsets[track_id]
                instances.append(
                    TrackedInstance(
                        global_track_id=track_id,
                        semantic_label=str(group.semantic_label),
                        points_world=packed_gpu[begin:end_offset],
                    )
                )
        else:
            adapter_prepare_ms = 1000.0 * (
                time.perf_counter() - adapter_prepare_started
            )
            adapter_pack_submit_ms = 0.0
            packed_gpu = torch.empty((0, 3), device=self.device, dtype=torch.float32)
            packed_ids_gpu = torch.empty((0,), device=self.device, dtype=torch.int32)

        first = results[0]
        adapter_total_ms = 1000.0 * (time.perf_counter() - adapter_started)
        tracker_timings = {str(key): float(value) for key, value in first.timings_ms.items()}
        tracker_timings.update(
            {
                "adapter_get_groups": float(adapter_get_groups_ms),
                "adapter_prepare": float(adapter_prepare_ms),
                "adapter_pack_submit": float(adapter_pack_submit_ms),
                "adapter_total": float(adapter_total_ms),
            }
        )
        return TrackedInstanceFrame(
            frame_index=int(first.frame.frame_index),
            stamp_ns=int(first.frame.timestamp_ns),
            instances=instances,
            packed_points_world=packed_gpu,
            packed_track_ids=packed_ids_gpu,
            track_ids=track_ids,
            track_offsets=track_offsets,
            view_results={str(result.frame.camera_name): result for result in results},
            tracker_timings_ms=tracker_timings,
            tracker_metadata=dict(first.metadata),
        )

    def process(self, frame: MultiCameraFrame) -> TrackedInstanceFrame | None:
        view_inputs, rgbs = self._view_inputs(frame)

        if self._prewarm_pending:
            self._run_owner(self.component.prewarm_tracker, rgbs)
            self._prewarm_pending = False
            return None

        if not self._initialized:
            initial_frames = self._run_owner(
                self.component.make_frames_batch,
                view_inputs,
            )
            initial_sam3 = self.sam3_worker.run_blocking(
                frame_index=int(initial_frames[0].frame_index),
                reference_frames=initial_frames,
            )
            results = self._run_owner(
                self.component.initialize_frames_batch,
                initial_frames,
                initial_sam3.detections_per_view,
                sam3_wall_ms=initial_sam3.wall_ms,
                sam3_filter_ms=initial_sam3.filter_cpu_ms,
                sam3_counts_per_view=initial_sam3.detections_per_class,
            )
            self._initialized = True
        else:
            correction = self.sam3_worker.poll()
            results = self._run_owner(
                self.component.process_arrays_batch,
                view_inputs,
                correction=correction,
            )

        if results and bool(results[0].metadata.get("sam3_refresh_due", False)):
            reference_frames = [result.frame for result in results]
            fallback_masks = self._run_owner(
                self.component.fallback_masks_from_results,
                results,
            )
            submitted = self.sam3_worker.submit(
                frame_index=int(reference_frames[0].frame_index),
                reference_frames=reference_frames,
                fallback_masks_per_view=fallback_masks,
            )
            if submitted:
                self._run_owner(
                    self.component.mark_sam3_submitted,
                    int(reference_frames[0].frame_index),
                )

        return self._to_tracked_frame(results)

    def close(self) -> None:
        # Stop SAM3 first so no correction can still hold tracker reference data.
        self.sam3_worker.close()
        try:
            self._run_owner(self.component.close)
        finally:
            self._owner.shutdown(wait=True, cancel_futures=False)
