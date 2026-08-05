from __future__ import annotations

import importlib
from pathlib import Path
import sys
from typing import Any

import torch

from .config import PipelineConfig
from .data_types import FlowInput, FlowResult


class DifFlowPredictor:
    """Step 5B: wrapper around the modified DifFlow3D streaming runner.

    The CUDA Graph runner is captured only once. It is captured with dt_s=1.0,
    and physical velocity is computed from the predicted displacement using the
    actual timestamp interval outside the graph. This avoids graph recapture
    when ROS timestamps vary slightly between frames.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.enabled = config.flow.enabled
        self.runner = None
        self._cached_target_stamp_ns: int | None = None

        if not self.enabled:
            return

        repo_path = Path(config.flow.repo_path).expanduser().resolve()
        checkpoint_path = Path(config.flow.checkpoint).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))

        module = importlib.import_module(config.flow.model_module)
        model_class = getattr(module, "PointConvBidirection")
        runner_class = getattr(
            module,
            "DifFlow3DStreamingCudaGraphRunner",
        )
        configure_fast = getattr(
            module,
            "configure_fast_inference",
            None,
        )
        if configure_fast is not None:
            configure_fast(config.flow.enable_tf32)

        model = model_class(iters=config.flow.iterations)
        checkpoint: Any = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        state_dict = self._extract_state_dict(checkpoint)
        state_dict = self._strip_module_prefix(state_dict)
        incompatible = model.load_state_dict(
            state_dict,
            strict=config.flow.strict_checkpoint,
        )
        self.missing_keys = tuple(incompatible.missing_keys)
        self.unexpected_keys = tuple(incompatible.unexpected_keys)

        model.to(self.device).eval()
        self.model = model
        self.runner_class = runner_class

    @staticmethod
    def _extract_state_dict(
        checkpoint: Any,
    ) -> dict[str, torch.Tensor]:
        if not isinstance(checkpoint, dict):
            raise TypeError("DifFlow3D checkpoint must be a mapping.")
        for key in ("state_dict", "model_state_dict", "model"):
            nested = checkpoint.get(key)
            if isinstance(nested, dict) and nested:
                return nested
        return checkpoint

    @staticmethod
    def _strip_module_prefix(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if state_dict and all(
            key.startswith("module.")
            for key in state_dict
        ):
            return {
                key.removeprefix("module."): value
                for key, value in state_dict.items()
            }
        return state_dict

    def _ensure_runner(self) -> None:
        if self.runner is not None:
            return

        # The graph predicts displacement. Keep graph dt fixed and divide the
        # displacement by the actual frame interval after replay.
        self.runner = self.runner_class(
            self.model,
            batch_size=1,
            num_points=self.config.flow.target_points,
            uncertainty=self.config.flow.uncertainty,
            warmup=self.config.flow.cuda_graph_warmup,
            enable_tf32=self.config.flow.enable_tf32,
            dt_s=1.0,
        )

    def reset(self) -> None:
        if self.runner is not None:
            self.runner.reset()
        self._cached_target_stamp_ns = None

    def predict(
        self,
        flow_input: FlowInput,
        previous_stamp_ns: int,
        current_stamp_ns: int,
    ) -> FlowResult:
        if flow_input.dt_s <= 0.0:
            raise ValueError(
                f"Flow input dt_s must be positive, got "
                f"{flow_input.dt_s:.9f}s."
            )

        if not self.enabled:
            zeros = torch.zeros_like(flow_input.previous_anchors)
            return FlowResult(
                source_anchors=flow_input.previous_anchors,
                warped_anchors=flow_input.previous_anchors,
                anchor_velocity=zeros,
            )

        self._ensure_runner()
        source_is_cached = (
            self._cached_target_stamp_ns == previous_stamp_ns
        )

        with torch.inference_mode():
            if not source_is_cached:
                self.runner.reset()
                self.runner.next_input.copy_(
                    flow_input.previous_anchors[None],
                    non_blocking=True,
                )
                if self.runner.replay_next() is not None:
                    raise RuntimeError(
                        "First streaming frame must only buffer."
                    )

            self.runner.next_input.copy_(
                flow_input.current_anchors[None],
                non_blocking=True,
            )
            if self.runner.replay_next() is None:
                raise RuntimeError(
                    "DifFlow3D did not produce a pair output."
                )

            displacement = self.runner.flow()[0]
            result = FlowResult(
                source_anchors=self.runner.source_points()[0],
                warped_anchors=self.runner.warped_points()[0],
                anchor_velocity=displacement / flow_input.dt_s,
            )

        self._cached_target_stamp_ns = current_stamp_ns
        return result
