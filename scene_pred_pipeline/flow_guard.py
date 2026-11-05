"""Guard against spurious large anchor displacements from segmentation bleed.

The online segmentation occasionally puts a patch of background into a person's mask for one frame. Those source points
have no counterpart in the current frame, and DifFlow3D pulls them onto some other surface: displacements of tens of
centimetres in one frame (real2: >100 mm in 1.7 % of frames, up to 634 mm). Nearly all of them (94-100 % above 150 mm)
share one signature that genuine fast motion does not have (0.75 % of reliable fast points): after warping by the
predicted displacement there is no current-frame point within ``support_radius_m``.

An anchor is flagged when its displacement exceeds ``min_displacement_m`` and its warped position has no current point
within ``support_radius_m``. The anchor keeps its source position (it is real geometry); its displacement and covariance
are replaced by the Gaussian-softmax interpolation of the UNFLAGGED anchors around its source position, with the deployed
recovery kernel (same sigma, cut-off and same-track conditioning). Flagged anchors are excluded from that interpolation by
giving them a track id no query uses. Where no unflagged anchor lies inside the recovery cut-off the displacement becomes
zero and the covariance is kept. Warped positions and the calibration scale follow the corrected displacement.
Everything stays on the GPU without host synchronisation.
"""
from __future__ import annotations

import dataclasses
from typing import Callable

import torch

from .data_types import FlowResult
from .velocity_recovery import pack_symmetric_upper, unpack_symmetric_upper

# Track id given to flagged anchors: no query point carries it, so the track-aware recovery ignores them.
_EXCLUDED_TRACK = -1_000_003


def _nearest_distance(queries: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Distance [M] from every query [M, 3] to its nearest point in ``points`` [N, 3] (exact, CUDA kNN).

    Call without grad mode: with grad enabled knn_point falls back to a dense [M, N] distance matrix by
    matmul (TF32 when enabled: errors up to ~28 mm on real frames) + topk, ~17x slower at 30k points."""
    from difflow3d.model.pointconv import knn_point

    if points.shape[0] == 0:
        return torch.full((queries.shape[0],), float("inf"), device=queries.device)
    idx = knn_point(1, points.float().unsqueeze(0).contiguous(), queries.float().unsqueeze(0).contiguous())[0, :, 0]
    return torch.linalg.vector_norm(queries.float() - points.float().index_select(0, idx.long()), dim=1)


class FlowGuard:
    """Corrects flagged anchors of a FlowResult (see module docstring)."""

    def __init__(
        self,
        recoverer,
        *,
        min_displacement_m: float,
        support_radius_m: float,
        calibration_scale: Callable[[torch.Tensor], torch.Tensor] | None = None,
    ) -> None:
        # The interpolation is always track-aware (flagged anchors are excluded through their track id, also
        # when recovery.restrict_same_track is false), which needs the local hash-grid backend.
        if getattr(recoverer, "backend", "local") != "local":
            raise ValueError("difflow.flow_guard needs difflow.recovery.backend: local")
        self.recoverer = recoverer
        self.min_displacement_m = float(min_displacement_m)
        self.support_radius_m = float(support_radius_m)
        self.calibration_scale = calibration_scale
        self.last_flagged: torch.Tensor | None = None     # bool [M] of the last call (device tensor, no sync)

    def _interpolate(self, anchors, field, track_ids, excluded_ids):
        rec = self.recoverer.recover(
            query_points=anchors,
            anchor_points=anchors,
            anchor_flow=field.contiguous(),
            dt_s=1.0,
            query_track_ids=track_ids,
            anchor_track_ids=excluded_ids,
        )
        return rec.flow, rec.local_neighbor_counts

    @torch.no_grad()
    def __call__(self, result: FlowResult, current_points: torch.Tensor) -> FlowResult:
        anchors = result.source_anchors.float()
        flow = result.anchor_flow.float()
        track_ids = result.source_anchor_track_ids
        support = _nearest_distance(anchors + flow, current_points)
        flagged = (torch.linalg.vector_norm(flow, dim=1) > self.min_displacement_m) & (support > self.support_radius_m)
        self.last_flagged = flagged
        excluded_ids = torch.where(flagged, torch.full_like(track_ids, _EXCLUDED_TRACK), track_ids)

        new_flow, counts = self._interpolate(anchors, flow, track_ids, excluded_ids)
        has_neighbor = counts > 0
        replacement = torch.where(has_neighbor[:, None], new_flow, torch.zeros_like(new_flow))
        corrected_flow = torch.where(flagged[:, None], replacement, flow)

        cov = result.anchor_flow_cov
        corrected_cov = cov
        if cov is not None:
            upper = pack_symmetric_upper(cov)
            parts = [self._interpolate(anchors, upper[:, k:k + 3], track_ids, excluded_ids)[0] for k in (0, 3)]
            new_cov = unpack_symmetric_upper(torch.cat(parts, dim=1))
            use_new = (flagged & has_neighbor)[:, None, None]
            corrected_cov = torch.where(use_new, new_cov.to(cov.dtype), cov)

        scale = result.anchor_flow_cov_scale
        if scale is not None and self.calibration_scale is not None:
            scale = torch.where(flagged, self.calibration_scale(corrected_flow).to(scale.dtype), scale)

        return dataclasses.replace(
            result,
            anchor_flow=corrected_flow.to(result.anchor_flow.dtype),
            warped_anchors=(result.source_anchors.float() + corrected_flow).to(result.warped_anchors.dtype),
            anchor_flow_cov=corrected_cov,
            anchor_flow_cov_scale=scale,
        )
