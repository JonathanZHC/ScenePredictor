from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair


class VelocityRecovery:
    """Dense current-frame velocity using DifFlow3D's world-space recoverer.

    The numerical recovery method is configured only in difflow.yaml. The one
    ScenePredictor-specific policy retained here is same-track conditioning,
    which prevents velocity mixing between nearby independently moving objects.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

        repo_path = Path(config.flow.repo_path).expanduser().resolve()
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        runtime_module = importlib.import_module("difflow3d.runtime")
        recoverer_class = getattr(runtime_module, "SoftmaxAnchorMotionRecoverer")

        recovery = config.difflow.recovery
        self.recoverer = recoverer_class(
            chunk_size=int(recovery.chunk_size),
            softmax_sigma_m=float(recovery.softmax_sigma_m),
            backend=str(recovery.backend),
            local_radius_sigma=float(recovery.local_radius_sigma),
            local_hash_size_factor=float(recovery.local_hash_size_factor),
        )

    def recover(
        self,
        pair: InstancePair,
        flow_result: FlowResult,
    ) -> torch.Tensor:
        """Recover every current-frame point in one CUDA call.

        When same-track conditioning is enabled, query and anchor track IDs are
        passed into DifFlow3D's local CUDA kernel. The kernel rejects anchors
        from other instances inside the hash-grid traversal, avoiding the old
        per-track boolean gathers, repeated hash builds, and repeated launches.
        """
        current = pair.current_points
        if current.shape[0] == 0:
            return torch.empty_like(current)

        kwargs = {}
        if self.config.recovery.restrict_same_track:
            if self.recoverer.backend != "local":
                raise ValueError(
                    "restrict_same_track requires difflow recovery.backend=local "
                    "for the fused track-aware CUDA recovery path"
                )
            kwargs = {
                "query_track_ids": pair.current_track_ids,
                "anchor_track_ids": flow_result.source_anchor_track_ids,
            }

        return self.recoverer.recover(
            query_points=current,
            anchor_points=flow_result.warped_anchors,
            anchor_flow=flow_result.anchor_flow,
            dt_s=float(pair.dt_s),
            **kwargs,
        ).velocity
