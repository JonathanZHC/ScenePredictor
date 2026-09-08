from __future__ import annotations

import time

import torch

from .config import PipelineConfig
from .data_types import (
    MultiCameraFrame,
    SceneVelocityOutput,
    TrackedInstanceFrame,
)
from .flow_prediction import DifFlowPredictor
from .instance_filter import CommonInstanceFilter
from .profiler import CycleProfiler
from .scene_assembly import SceneCloudAssembler
from .tracker_adapter import MultiViewTrackerAdapter
from .velocity_recovery import VelocityRecovery


class ScenePredictionPipeline:
    """MultiViewRGBDTracker -> common-ID DifFlow3D -> dense velocity recovery.

    The numerical pipeline intentionally does not build ROS visualization data.
    Raw tracker result references are carried to RosVisualizer, which lazily
    materializes overlays/masks/clouds only when a subscriber is connected.
    """

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
        # Full-scene cloud (tracked points + velocity, rest points + zero velocity)
        # for in-process consumers such as the safety filter.
        self.scene_assembler = (
            SceneCloudAssembler(config, tracker_config=self.tracker.tracker_config)
            if config.scene_cloud.enabled
            else None
        )
        self.profiler = CycleProfiler(config.runtime.enable_cuda_timing)
        self.previous_tracked: TrackedInstanceFrame | None = None
        self.last_flow_gap_s: float | None = None

    @property
    def device(self) -> torch.device:
        return torch.device(self.config.runtime.device)

    def _empty_points(self) -> torch.Tensor:
        # Immutable zero-length tensors: allocate once instead of 6-9 per frame.
        cached = getattr(self, "_empty_points_cache", None)
        if cached is None:
            cached = torch.empty((0, 3), device=self.device, dtype=torch.float32)
            self._empty_points_cache = cached
        return cached

    def _empty_ids(self) -> torch.Tensor:
        cached = getattr(self, "_empty_ids_cache", None)
        if cached is None:
            cached = torch.empty((0,), device=self.device, dtype=torch.int32)
            self._empty_ids_cache = cached
        return cached

    def _all_tracked_points(
        self,
        tracked: TrackedInstanceFrame,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # These are already one contiguous CUDA allocation per frame.  Do not
        # rebuild them from per-instance views with torch.cat()/torch.full().
        return tracked.packed_points_world, tracked.packed_track_ids

    def _record_tracker_breakdown(self, tracked: TrackedInstanceFrame) -> None:
        values = tracked.tracker_timings_ms
        tracking_model = sum(
            float(values.get(name, 0.0))
            for name in (
                "tracker_reinit",
                "tracker_propagate",
                "tracker_direct_correction",
            )
        )
        self.profiler.record("tracking_model", tracking_model)
        self.profiler.record("postprocess", float(values.get("postprocess_total", 0.0)))
        self.profiler.record("alignment", float(values.get("alignment_total", 0.0)))
        self.profiler.record("adapter", float(values.get("adapter_total", 0.0)))

        # These values are useful diagnostics, but SAM3 inference/filtering runs on
        # its separate async worker and must never be visually added to cycle_total.
        for name in ("sam3_async", "sam3_filter", "sam3_slot_assoc"):
            if name in values:
                self.profiler.record(name, float(values[name]))

    def _empty_output(
        self,
        frame: MultiCameraFrame,
        tracked: TrackedInstanceFrame | None,
    ) -> SceneVelocityOutput:
        tracked_points = self._empty_points()
        tracked_ids = self._empty_ids()
        view_results = {}
        if tracked is not None:
            tracked_points, tracked_ids = self._all_tracked_points(tracked)
            view_results = tracked.view_results
        timings = self.profiler.finish()
        output = SceneVelocityOutput(
            stamp_ns=int(frame.stamp_ns if tracked is None else tracked.stamp_ns),
            flow_dt_s=0.0,
            tracked_points=tracked_points,
            tracked_track_ids=tracked_ids,
            flow_points=self._empty_points(),
            flow_velocity=self._empty_points(),
            flow_track_ids=self._empty_ids(),
            source_anchors=self._empty_points(),
            warped_anchors=self._empty_points(),
            removed_outlier_points=self._empty_points(),
            view_results=view_results,
            common_track_ids=(),
            flow_valid=False,
            timings_ms=timings,
        )
        self._assemble_scene(output)
        return output

    def _assemble_scene(self, output: SceneVelocityOutput) -> None:
        if self.scene_assembler is None:
            return
        started = time.perf_counter()
        self.scene_assembler.assemble(output)
        self.profiler.record_async("scene_assembly", 1000.0 * (time.perf_counter() - started))

    def process(self, frame: MultiCameraFrame) -> SceneVelocityOutput:
        self.profiler.start_cycle()
        self.last_flow_gap_s = None

        with self.profiler.stage("tracker_total", cuda=False):
            current = self.tracker.process(frame)

        # The first synchronized bundle may be consumed solely by EfficientTAM
        # prewarm and is intentionally not a temporal source.
        if current is None:
            return self._empty_output(frame, None)

        self._record_tracker_breakdown(current)

        tracked_points, tracked_ids = self._all_tracked_points(current)

        previous = self.previous_tracked
        # Always advance to the immediately current tracker frame, even when the
        # pair is empty, stale, or skipped.  This also keeps the packed source
        # CUDA storage alive for the next zero-copy common-instance selection.
        self.previous_tracked = current

        with self.profiler.stage("instance_filter", cuda=True):
            pair = self.instance_filter.select(previous, current)

        flow_points = self._empty_points()
        flow_velocity = self._empty_points()
        flow_track_ids = self._empty_ids()
        flow_dt_s = 0.0
        source_anchors = self._empty_points()
        warped_anchors = self._empty_points()
        common_ids: tuple[int, ...] = ()
        flow_valid = False
        outlier_filter_info: dict[str, object] = {}
        outlier_debug: dict[str, object] = {}
        removed_outlier_points = self._empty_points()
        dense_filter_counts: tuple[int, int, int] | None = None

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
                with self.profiler.stage("difflow_total", cuda=True):
                    # track_labels feed only the detailed outlier debug report.
                    track_labels = None
                    if self.flow_predictor.detailed_outlier_output:
                        track_labels = {
                            int(instance.global_track_id): instance.semantic_label
                            for instance in current.instances
                        }
                    flow_result = self.flow_predictor.predict(pair, track_labels=track_labels)

                # Boolean-mask indexing on CUDA runs nonzero() and syncs the host
                # to size the result. With the outlier filter disabled the keep
                # mask is all-ones by construction, so alias the inputs (zero
                # syncs, zero copies). When enabled, resolve the indices once and
                # share them with recovery (1 sync instead of 5).
                total_points = int(pair.current_points.shape[0])
                if not self.flow_predictor.outlier_filter_enabled:
                    flow_points = pair.current_points
                    flow_track_ids = pair.current_track_ids
                    dense_filter_counts = (total_points, total_points, 0)
                    removed_outlier_points = self._empty_points()
                else:
                    target_keep = flow_result.target_input_keep_mask
                    keep_index = target_keep.nonzero(as_tuple=True)[0]
                    flow_points = pair.current_points.index_select(0, keep_index)
                    flow_track_ids = pair.current_track_ids.index_select(0, keep_index)
                    kept = int(keep_index.shape[0])
                    dense_filter_counts = (total_points, kept, total_points - kept)
                    if self.config.output.publish_removed_outlier_points:
                        removed_index = (~target_keep).nonzero(as_tuple=True)[0]
                        removed_outlier_points = pair.current_points.index_select(0, removed_index)

                with self.profiler.stage("velocity_recovery", cuda=True):
                    flow_velocity = self.recovery.recover(
                        pair, flow_result, query_points=flow_points, query_track_ids=flow_track_ids
                    )
                flow_dt_s = float(pair.dt_s)
                source_anchors = flow_result.source_anchors
                warped_anchors = flow_result.warped_anchors
                flow_valid = True

        # Visualization is deliberately absent from the numerical critical path.
        # RosVisualizer checks subscription counts before any D2H conversion,
        # overlay construction, mask merge, PointCloud2 packing, or marker build.
        timings = self.profiler.finish()
        if flow_valid:
            runner_timings, outlier_filter_info, outlier_debug = (
                self.flow_predictor.resolve_after_cycle_fence()
            )
            if dense_filter_counts is not None:
                (
                    outlier_filter_info["dense_input_points"],
                    outlier_filter_info["dense_retained_points"],
                    outlier_filter_info["dense_removed_points"],
                ) = dense_filter_counts
            timings.update(runner_timings)
            timings["difflow_other"] = self.profiler.record_difflow_timings(
                runner_timings,
                timings["difflow_total"],
            )
            self.profiler.record_outlier_filter(outlier_filter_info)
        output = SceneVelocityOutput(
            stamp_ns=int(current.stamp_ns),
            flow_dt_s=flow_dt_s,
            tracked_points=tracked_points,
            tracked_track_ids=tracked_ids,
            flow_points=flow_points,
            flow_velocity=flow_velocity,
            flow_track_ids=flow_track_ids,
            source_anchors=source_anchors,
            warped_anchors=warped_anchors,
            removed_outlier_points=removed_outlier_points,
            view_results=current.view_results,
            common_track_ids=common_ids,
            flow_valid=flow_valid,
            timings_ms=timings,
            outlier_filter_info=outlier_filter_info,
            outlier_debug=outlier_debug,
        )
        self._assemble_scene(output)
        return output

    def close(self) -> None:
        self.tracker.close()
