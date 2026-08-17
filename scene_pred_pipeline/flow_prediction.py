from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair


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
            auto_spatial_scale=bool(prep.auto_spatial_scale),
            target_model_volume=float(prep.target_model_volume),
            fixed_spatial_scale=float(prep.fixed_spatial_scale),
            final_selection=str(prep.final_selection),
            enable_profiling=False,
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

    def reset_calibration(self) -> None:
        """Reset temporal state and one-shot DifFlow preprocessing calibration."""
        if self.runner is not None:
            self.runner.reset_preprocess_calibration()
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None

    def predict(self, pair: InstancePair) -> FlowResult:
        if pair.dt_s <= 0.0:
            raise ValueError(f"Flow dt_s must be positive, got {pair.dt_s:.9f}s")

        self._ensure_runner()
        signature = tuple(int(value) for value in pair.common_track_ids)
        source_is_cached = (
            self._cached_target_stamp_ns == int(pair.previous_stamp_ns)
            and self._cached_track_signature == signature
        )

        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                self.runner.stage_world(
                    pair.previous_points,
                    point_ids=pair.previous_track_ids,
                )
                if self.runner.replay_next() is not None:
                    raise RuntimeError("First streaming frame must only buffer")

            self.runner.stage_world(
                pair.current_points,
                point_ids=pair.current_track_ids,
            )
            if self.runner.replay_next() is None:
                raise RuntimeError("DifFlow3D did not produce a pair output")

            anchor_flow = self.runner.flow_world()[0]
            source_anchors = self.runner.source_points_world()[0]
            warped_anchors = self.runner.warped_points_world()[0]
            source_anchor_ids = self.runner.source_point_ids()

            result = FlowResult(
                source_anchors=source_anchors,
                warped_anchors=warped_anchors,
                anchor_flow=anchor_flow,
                source_anchor_track_ids=source_anchor_ids,
            )

        self._cached_target_stamp_ns = int(pair.current_stamp_ns)
        self._cached_track_signature = signature
        return result
