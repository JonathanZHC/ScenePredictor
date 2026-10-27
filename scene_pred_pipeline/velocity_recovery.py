from __future__ import annotations

import importlib
from pathlib import Path
import sys

import torch

from .config import PipelineConfig
from .data_types import FlowResult, InstancePair


def anchor_velocity_region(cov: torch.Tensor, scale: torch.Tensor, dt_s: float, region: str) -> torch.Tensor:
    """Calibrated velocity region R [M, 3, 3] (m^2/s^2) of each point (anchor or dense point) from its one-pair
    displacement covariance Sigma [M, 3, 3] (m^2) and calibration scale s [M]: the true velocity lies in
    {v + d : d^T R^-1 d <= 1} with the calibrated probability. ellipsoid: R = s^2 Sigma / dt^2;
    ball: R = (s det(Sigma)^(1/6) / dt)^2 I."""
    factor = (scale.float() / float(dt_s)) ** 2
    if region == "ellipsoid":
        return cov.float() * factor[:, None, None]
    radius2 = torch.linalg.det(cov.float()).clamp_min(0.0) ** (1.0 / 3.0)
    eye = torch.eye(3, dtype=cov.dtype if cov.is_floating_point() else torch.float32, device=cov.device)
    return (factor * radius2)[:, None, None] * eye


_UPPER = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


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
        *,
        query_points: torch.Tensor | None = None,
        query_track_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Recover every retained current-frame point in one CUDA call.

        When same-track conditioning is enabled, query and anchor track IDs are
        passed into DifFlow3D's local CUDA kernel. The kernel rejects anchors
        from other instances inside the hash-grid traversal, avoiding the old
        per-track boolean gathers, repeated hash builds, and repeated launches.

        ``query_points`` / ``query_track_ids`` are the already-retained current
        points (the caller resolves the keep mask once). When omitted they are
        derived from ``flow_result.target_input_keep_mask`` (one host sync).
        """
        if query_points is None or query_track_ids is None:
            keep_mask = flow_result.target_input_keep_mask
            if keep_mask.shape != (pair.current_points.shape[0],):
                raise RuntimeError("DifFlow target keep mask does not match the input cloud")
            keep_index = keep_mask.nonzero(as_tuple=True)[0]
            query_points = pair.current_points.index_select(0, keep_index)
            query_track_ids = pair.current_track_ids.index_select(0, keep_index)
        current = query_points
        if current.shape[0] == 0:
            return torch.empty_like(current)

        return self.recoverer.recover(
            query_points=current,
            anchor_points=flow_result.warped_anchors,
            anchor_flow=flow_result.anchor_flow,
            dt_s=float(pair.dt_s),
            **self._track_kwargs(flow_result, query_track_ids),
        ).velocity

    def recover_region(
        self,
        flow_result: FlowResult,
        anchor_region: torch.Tensor,
        *,
        query_points: torch.Tensor,
        query_track_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Per-anchor matrices [M, 3, 3] (e.g. Sigma) carried to the query points [Nq, 3, 3] with the same
        Gaussian-softmax weights (and same-track conditioning) as the velocity: two recoveries of the six unique
        entries. A convex combination of symmetric positive semi-definite matrices stays one."""
        if query_points.shape[0] == 0:
            return query_points.new_zeros((0, 3, 3))
        upper = torch.stack([anchor_region[:, i, j] for i, j in _UPPER], dim=1).float()
        kwargs = self._track_kwargs(flow_result, query_track_ids)
        parts = [
            self.recoverer.recover(
                query_points=query_points,
                anchor_points=flow_result.warped_anchors,
                anchor_flow=upper[:, k:k + 3].contiguous(),
                dt_s=1.0,
                **kwargs,
            ).flow
            for k in (0, 3)
        ]
        values = torch.cat(parts, dim=1)
        region = query_points.new_empty((query_points.shape[0], 3, 3))
        for k, (i, j) in enumerate(_UPPER):
            region[:, i, j] = values[:, k]
            region[:, j, i] = values[:, k]
        return region

    def _track_kwargs(self, flow_result: FlowResult, query_track_ids: torch.Tensor) -> dict:
        if not self.config.recovery.restrict_same_track:
            return {}
        if self.recoverer.backend != "local":
            raise ValueError(
                "restrict_same_track requires difflow recovery.backend=local "
                "for the fused track-aware CUDA recovery path"
            )
        return {"query_track_ids": query_track_ids, "anchor_track_ids": flow_result.source_anchor_track_ids}
