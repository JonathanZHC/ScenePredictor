from __future__ import annotations

import importlib
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair


_NEIGHBOR_OFFSETS = tuple(
    (dx, dy, dz)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    for dz in (-1, 0, 1)
    if (dx, dy, dz) != (0, 0, 0)
)


class DifFlowPredictor:
    """Always-on adapter for DifFlow3D's production streaming runner.

    ScenePredictor passes the tracker's first-downsampled world cloud directly
    into DifFlow. DifFlow owns adaptive voxel-2 reduction, exact-count selection,
    frozen spatial scaling, CUDA-Graph inference, and world-space anchor outputs.

    Encoder reuse is valid only when the previous source is exactly the target
    buffered by the last pair. Both timestamp and common-ID signature are used
    because a changing instance intersection changes the combined input cloud.
    """

    def __init__(
        self,
        config: PipelineConfig,
        *,
        upstream_voxel_size_m: float,
    ) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.upstream_voxel_size_m = float(upstream_voxel_size_m)
        if self.upstream_voxel_size_m <= 0.0:
            raise ValueError("upstream_voxel_size_m must be positive")
        self.runner = None
        self._cached_target_stamp_ns: int | None = None
        self._cached_track_signature: tuple[int, ...] | None = None
        self._last_target_preprocess_info: dict[str, object] | None = None
        self._debug_pending_current: dict[str, object] | None = None

        if self.device.type != "cuda":
            raise ValueError("DifFlow3D deployment inference requires CUDA")

        difflow_device = torch.device(config.difflow.runtime.device)
        if difflow_device.type != "cuda":
            raise ValueError("difflow.runtime.device must be CUDA for deployment")
        if (
            difflow_device.index is not None
            and self.device.index is not None
            and difflow_device.index != self.device.index
        ):
            raise ValueError(
                "ScenePredictor and DifFlow must use the same CUDA device: "
                f"runtime.device={config.runtime.device!r}, "
                f"difflow.runtime.device={config.difflow.runtime.device!r}"
            )

        repo_path = Path(config.flow.repo_path).expanduser().resolve()
        if not repo_path.is_dir():
            raise FileNotFoundError(f"DifFlow3D repository not found: {repo_path}")
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        model_module = importlib.import_module("difflow3d.model")
        runtime_module = importlib.import_module("difflow3d.runtime")

        self.runner_class = getattr(
            runtime_module,
            "DifFlow3DStreamingCudaGraphRunner",
        )
        configure_fast = getattr(runtime_module, "configure_fast_inference")
        load_checkpoint = getattr(runtime_module, "load_checkpoint")
        self.resolve_outlier_statistics = getattr(
            runtime_module,
            "resolve_voxel_outlier_statistics",
        )
        model_class = getattr(model_module, "PointConvBidirection")

        difflow = config.difflow
        configure_fast(difflow.runtime.enable_tf32)
        iterations = difflow.model.iterations
        model = model_class(
            iters=max(iterations.coarse, iterations.middle, iterations.fine),
            coarse_iters=iterations.coarse,
            middle_iters=iterations.middle,
            fine_iters=iterations.fine,
        )

        checkpoint = Path(difflow.model.checkpoint).expanduser()
        if not checkpoint.is_absolute():
            checkpoint = repo_path / checkpoint
        checkpoint = checkpoint.resolve()

        self.checkpoint_report = load_checkpoint(
            model,
            checkpoint,
            strict=difflow.model.strict_checkpoint,
        )
        self.missing_keys = tuple(self.checkpoint_report.missing_keys)
        self.unexpected_keys = tuple(self.checkpoint_report.unexpected_keys)

        self.model = model.to(self.device).eval()
        if difflow.model.disable_bn_running_stats:
            for layer in self.model.modules():
                if isinstance(layer, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                    layer.track_running_stats = False

    def _ensure_runner(self) -> None:
        if self.runner is not None:
            return

        difflow = self.config.difflow
        prep = difflow.preprocessing
        spatial_scale = prep.auto_spatial_scale
        outlier_filter = prep.outlier_filter
        self.runner = self.runner_class(
            self.model,
            batch_size=1,
            num_points=int(prep.fps_points),
            uncertainty=float(difflow.model.uncertainty),
            warmup=int(difflow.runtime.cuda_graph_warmup),
            enable_tf32=bool(difflow.runtime.enable_tf32),
            # Decode predicts displacement. ScenePredictor divides by the true
            # timestamp delta outside the graph, avoiding graph recapture when
            # frame timing jitters around the nominal sensor rate.
            dt_s=1.0,
            second_base_voxel_size_m=self.upstream_voxel_size_m,
            second_candidate_ratio=float(prep.second_candidate_ratio),
            auto_spatial_scale=bool(spatial_scale.enable),
            target_model_volume=float(spatial_scale.target_model_volume),
            fixed_spatial_scale=float(spatial_scale.fixed_spatial_scale),
            final_selection=str(prep.final_selection),
            outlier_filter_enabled=bool(outlier_filter.enabled),
            outlier_filter_min_component_size_ratio=float(
                outlier_filter.min_component_size_ratio
            ),
            enable_profiling=bool(self.config.runtime.enable_cuda_timing),
            validate_finite=bool(difflow.runtime.validate_finite),
        )

    def prepare(self) -> None:
        """Capture DifFlow CUDA graphs before tracker CUDA graphs are created."""
        torch.cuda.synchronize(self.device)
        self._ensure_runner()
        torch.cuda.synchronize(self.device)

    def reset(self) -> None:
        """Reset temporal reuse while keeping frozen voxel/scale calibration."""
        if self.runner is not None:
            self.runner.reset()
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None
        self._last_target_preprocess_info = None
        self._debug_pending_current = None

    def reset_calibration(self) -> None:
        """Reset temporal state and one-shot DifFlow preprocessing calibration."""
        if self.runner is not None:
            self.runner.reset_preprocess_calibration()
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None
        self._last_target_preprocess_info = None
        self._debug_pending_current = None

    @property
    def outlier_filter_enabled(self) -> bool:
        return bool(self.config.difflow.preprocessing.outlier_filter.enabled)

    @property
    def detailed_outlier_output(self) -> bool:
        return self._detailed_outlier_output

    @property
    def _detailed_outlier_output(self) -> bool:
        config = self.config.difflow.preprocessing.outlier_filter
        return bool(config.enabled and config.detailed_output)

    def _capture_debug_candidates(
        self,
        points: torch.Tensor,
        point_ids: torch.Tensor | None,
        *,
        stamp_ns: int,
        track_labels: dict[int, str],
    ) -> dict[str, object] | None:
        """Rebuild exact voxel-2 representatives only in detailed debug mode."""
        if not self._detailed_outlier_output:
            return None
        assert self.runner is not None
        preprocessor = self.runner.preprocessor
        if int(points.shape[0]) <= int(preprocessor.target_candidate_count):
            return None
        voxel_size = preprocessor.second_voxel_size_m
        if voxel_size is None:
            return None

        candidate_points, candidate_indices, _, absolute, *_ = (
            preprocessor._voxel_downsample(points, voxel_size, point_ids)
        )
        candidate_ids = (
            point_ids.index_select(0, candidate_indices).to(torch.int32)
            if point_ids is not None
            else torch.full(
                (candidate_points.shape[0],),
                -1,
                device=candidate_points.device,
                dtype=torch.int32,
            )
        )
        return {
            "stamp_ns": int(stamp_ns),
            "voxel_size_m": float(voxel_size),
            "points": candidate_points,
            "coords": absolute,
            "track_ids": candidate_ids.contiguous(),
            "track_labels": dict(track_labels),
        }

    @staticmethod
    def _connected_components(
        coords: np.ndarray,
        track_ids: np.ndarray,
    ) -> list[np.ndarray]:
        lookup = {
            (int(track_ids[index]), *tuple(map(int, coord))): index
            for index, coord in enumerate(coords)
        }
        visited = np.zeros((len(coords),), dtype=bool)
        components: list[np.ndarray] = []
        for seed in range(len(coords)):
            if visited[seed]:
                continue
            visited[seed] = True
            stack = [seed]
            members: list[int] = []
            while stack:
                index = stack.pop()
                members.append(index)
                x, y, z = map(int, coords[index])
                track_id = int(track_ids[index])
                for dx, dy, dz in _NEIGHBOR_OFFSETS:
                    neighbor = lookup.get(
                        (track_id, x + dx, y + dy, z + dz)
                    )
                    if neighbor is not None and not visited[neighbor]:
                        visited[neighbor] = True
                        stack.append(neighbor)
            components.append(np.asarray(members, dtype=np.int64))
        return components

    def _analyze_debug_candidates(
        self,
        pending: dict[str, object],
        outlier_info: dict[str, object],
    ) -> dict[str, object]:
        points = pending["points"].detach().float().cpu().numpy()
        coords = (
            pending["coords"].detach().cpu().numpy().astype(np.int32, copy=False)
        )
        track_ids = (
            pending["track_ids"].detach().cpu().numpy().astype(np.int32, copy=False)
        )
        track_labels = pending["track_labels"]
        assert isinstance(track_labels, dict)

        components = self._connected_components(coords, track_ids)
        largest = max((len(component) for component in components), default=0)
        filter_config = self.config.difflow.preprocessing.outlier_filter
        minimum_ratio = float(filter_config.min_component_size_ratio)

        reports: list[dict[str, object]] = []
        for block_id, component in enumerate(components):
            size = int(len(component))
            track_id = int(track_ids[component[0]])
            centroid = points[component].mean(axis=0)
            bbox_min = coords[component].min(axis=0)
            bbox_max = coords[component].max(axis=0)
            reports.append(
                {
                    "block_id": block_id,
                    "track_id": track_id,
                    "size_voxels": size,
                    "centroid_world": tuple(float(value) for value in centroid),
                    "bbox_min_voxel": tuple(int(value) for value in bbox_min),
                    "bbox_max_voxel": tuple(int(value) for value in bbox_max),
                    "removed": False,
                    "reason": "unclassified",
                }
            )

        reports_by_track: dict[int, list[dict[str, object]]] = {}
        for report in reports:
            reports_by_track.setdefault(int(report["track_id"]), []).append(report)

        instances_with_removed = 0
        for track_reports in reports_by_track.values():
            ordered = sorted(
                track_reports,
                key=lambda report: (
                    -int(report["size_voxels"]),
                    int(report["block_id"]),
                ),
            )
            maximum_size = int(ordered[0]["size_voxels"])
            threshold = maximum_size * minimum_ratio
            for index, report in enumerate(ordered):
                size = int(report["size_voxels"])
                keep = size > threshold
                report["object_block_rank"] = index + 1
                report["object_max_size_voxels"] = maximum_size
                report["relative_to_max"] = float(size / maximum_size)
                report["size_threshold_voxels"] = float(threshold)
                report["removed"] = not keep
                if not keep:
                    report["reason"] = "at_or_below_relative_size_threshold"
                elif index == 0:
                    report["reason"] = "largest_component"
                else:
                    report["reason"] = "above_relative_size_threshold"
            if any(bool(report["removed"]) for report in ordered):
                instances_with_removed += 1

        reconstructed_removed = sum(
            int(report["size_voxels"])
            for report in reports
            if bool(report["removed"])
        )
        actual_removed = int(outlier_info.get("removed_voxels", 0))
        core_statistics = outlier_info.get("statistics")
        if not isinstance(core_statistics, dict):
            core_statistics = {}
        actual_blocks = int(core_statistics.get("component_count", -1))
        actual_largest = int(
            core_statistics.get("largest_component_voxels", -1)
        )
        consistent = (
            reconstructed_removed == actual_removed
            and actual_blocks == len(reports)
            and actual_largest == largest
        )
        outlier_info["detailed_output"] = True
        outlier_info["detailed_block_count"] = len(reports)
        outlier_info["detailed_reconstruction_consistent"] = consistent

        stamp_ns = int(pending["stamp_ns"])
        voxel_size = float(pending["voxel_size_m"])
        lines = [
            "[DifFlow outlier detailed] "
            f"stamp_ns={stamp_ns} scope=per-track-26-connected "
            f"voxels={len(coords)} blocks={len(reports)} largest={largest} "
            f"voxel_size_m={voxel_size:.6f} "
            f"min_component_size_ratio={minimum_ratio:.4f}",
            "  Rule: for each object, keep a block only when "
            "size > object_max_size * min_component_size_ratio.",
            "  NOTE: track=-1 is the virtual object used when IDs are unavailable.",
        ]
        for report in reports:
            track_id = int(report["track_id"])
            label = str(track_labels.get(track_id, "untracked"))
            centroid = report["centroid_world"]
            lines.append(
                "  "
                f"block={int(report['block_id']):03d} "
                f"object={label!r}/track:{track_id} "
                f"object_rank={int(report['object_block_rank'])} "
                f"size={int(report['size_voxels'])} "
                f"object_max={int(report['object_max_size_voxels'])} "
                f"relative_to_max={float(report['relative_to_max']):.4f} "
                f"threshold={float(report['size_threshold_voxels']):.3f} "
                f"centroid=({centroid[0]:.3f},{centroid[1]:.3f},{centroid[2]:.3f}) "
                f"bbox={report['bbox_min_voxel']}..{report['bbox_max_voxel']} "
                f"decision={'REMOVE' if report['removed'] else 'KEEP'} "
                f"reason={report['reason']}"
            )
        lines.append(
            "  reconstruction_check="
            f"{'OK' if consistent else 'MISMATCH'} "
            f"removed={reconstructed_removed}/{actual_removed} "
            f"blocks={len(reports)}/{actual_blocks} "
            f"largest={largest}/{actual_largest} (debug/difflow)"
        )
        print("\n".join(lines), flush=True)

        return {
            "blocks": reports,
            "instances_with_removed_components": instances_with_removed,
        }

    def resolve_after_cycle_fence(
        self,
    ) -> tuple[dict[str, float], dict[str, object], dict[str, object]]:
        """Resolve runner timing and filter statistics after cycle synchronization."""
        if self.runner is None:
            return {}, {}, {}

        profile = self.runner.resolve_profile_window(synchronize=False)
        timing = {
            f"runner_{name.removesuffix('_ms')}": float(value)
            for name, value in profile.items()
        }

        info = dict(self._last_target_preprocess_info or {})
        self.resolve_outlier_statistics(info)
        statistics = info.get("outlier_filter_statistics")
        filter_config = self.config.difflow.preprocessing.outlier_filter
        outlier_info: dict[str, object] = {
            "enabled": bool(filter_config.enabled),
            "detailed_output": bool(filter_config.detailed_output),
            "applied": bool(info.get("outlier_filter_applied", False)),
            "input_voxels": int(info.get("outlier_filter_input_count", 0)),
            "retained_voxels": int(
                info.get("outlier_filter_retained_count", 0)
            ),
            "removed_voxels": int(info.get("outlier_filter_removed_count", 0)),
            "min_component_size_ratio": float(
                filter_config.min_component_size_ratio
            ),
            "statistics": dict(statistics) if isinstance(statistics, dict) else {},
        }
        debug: dict[str, object] = {}
        if self._detailed_outlier_output:
            debug_start = time.perf_counter()
            pending = self._debug_pending_current
            if pending is None:
                print(
                    "[DifFlow outlier detailed] voxel-2/filter bypassed; "
                    "no component cloud for this frame.",
                    flush=True,
                )
            else:
                debug = self._analyze_debug_candidates(
                    pending,
                    outlier_info,
                )
            self._debug_pending_current = None
            timing["outlier_debug"] = (time.perf_counter() - debug_start) * 1000.0
        return timing, outlier_info, debug

    def predict(
        self,
        pair: InstancePair,
        *,
        track_labels: dict[int, str] | None = None,
    ) -> FlowResult:
        if pair.dt_s <= 0.0:
            raise ValueError(f"Flow dt_s must be positive, got {pair.dt_s:.9f}s")

        self._ensure_runner()
        track_labels = dict(track_labels or {})
        signature = tuple(int(value) for value in pair.common_track_ids)
        source_is_cached = (
            self._cached_target_stamp_ns == int(pair.previous_stamp_ns)
            and self._cached_track_signature == signature
        )

        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                self.runner.begin_profile_window()
                self.runner.stage_world(
                    pair.previous_points,
                    point_ids=pair.previous_track_ids,
                )
                if self.runner.replay_next() is not None:
                    raise RuntimeError("First streaming frame must only buffer")
            else:
                self.runner.begin_profile_window()

            self.runner.stage_world(
                pair.current_points,
                point_ids=pair.current_track_ids,
            )
            self._debug_pending_current = self._capture_debug_candidates(
                pair.current_points,
                pair.current_track_ids,
                stamp_ns=pair.current_stamp_ns,
                track_labels=track_labels,
            )

            if self.runner.replay_next() is None:
                raise RuntimeError("DifFlow3D did not produce a pair output")

            result = FlowResult(
                source_anchors=self.runner.source_points_world()[0],
                warped_anchors=self.runner.warped_points_world()[0],
                anchor_flow=self.runner.flow_world()[0],
                source_anchor_track_ids=self.runner.source_point_ids(),
                target_input_keep_mask=self.runner.target_input_keep_mask(),
            )
            self._last_target_preprocess_info = (
                self.runner.target_preprocess_info()
            )

        self._cached_target_stamp_ns = int(pair.current_stamp_ns)
        self._cached_track_signature = signature
        return result
