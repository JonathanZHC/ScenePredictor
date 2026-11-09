"""Guard against spurious large anchor displacements from segmentation bleed.

The online segmentation occasionally puts a patch of background into a person's mask for one frame. Those source points
have no counterpart in the current frame, and DifFlow3D pulls them onto some other surface: displacements of tens of
centimetres in one frame (real2: >100 mm in 1.7 % of frames, up to 634 mm). Nearly all of them (94-100 % above 150 mm)
share one signature that genuine fast motion does not have (0.75 % of reliable fast points): after warping by the
predicted displacement there is no current-frame point within ``support_radius_m``.

An anchor is flagged when its displacement exceeds ``min_displacement_m`` and its warped position has no current point
within the support radius. The anchor keeps its source position (it is real geometry); its displacement and covariance
are replaced by the Gaussian-softmax interpolation of the UNFLAGGED anchors around its source position, with the deployed
recovery kernel (same sigma, cut-off and same-track conditioning). Flagged anchors are excluded from that interpolation by
giving them a track id no query uses. Warped positions and the calibration scale follow the corrected displacement.

2026-11-09 (a 3 m/s ball had its whole track zeroed): the guard is for a MINORITY of anchors at a mask border.
  - support radius = max(support_radius_m, support_spacing_factor x median nearest-neighbour spacing of the current cloud):
    with a 15 mm depth noise at 3 m and independent resampling a fixed 15 mm radius finds no support;
  - a track with more than ``majority_fraction`` of its anchors flagged is not bleed (fast object or a bad flow): its
    anchors keep their flow;
  - a flagged anchor without unflagged neighbours falls back to its track's centroid displacement (current-frame centroid
    minus anchor centroid of the same track), never to zero (zero turns a moving object into a static one).
Every call reports what it changed (``last_stats``, device tensor; ``report()`` prints at most every report_period_s).
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
        support_spacing_factor: float = 2.5,
        majority_fraction: float = 0.5,
        report_period_s: float = 1.0,
        log: Callable[[str], None] | None = None,
    ) -> None:
        # The interpolation is always track-aware (flagged anchors are excluded through their track id, also
        # when recovery.restrict_same_track is false), which needs the local hash-grid backend.
        if getattr(recoverer, "backend", "local") != "local":
            raise ValueError("difflow.flow_guard needs difflow.recovery.backend: local")
        self.recoverer = recoverer
        self.min_displacement_m = float(min_displacement_m)
        self.support_radius_m = float(support_radius_m)
        self.calibration_scale = calibration_scale
        self.support_spacing_factor = float(support_spacing_factor)
        self.majority_fraction = float(majority_fraction)
        self.report_period_s = float(report_period_s)
        self.log = log or (lambda msg: print(msg, flush=True))
        self.last_flagged: torch.Tensor | None = None     # bool [M] of the last call (device tensor, no sync)
        # [n_anchors, flagged_raw, kept_majority, replaced_neighbours, centroid_fallback, n_majority_tracks] (device tensor)
        self.last_stats: torch.Tensor | None = None
        self._last_report = 0.0
        self.calls = 0

    def stats_list(self) -> list[int]:
        """Host copy of last_stats (one sync); zeros when the guard did not run."""
        return [int(v) for v in self.last_stats.tolist()] if self.last_stats is not None else [0] * 6

    def report(self, force: bool = False) -> None:
        """Print the last call's changes if anything was changed, at most every report_period_s (one host sync)."""
        import time
        now = time.monotonic()
        if self.last_stats is None or (not force and now - self._last_report < self.report_period_s):
            return
        n, flagged, kept, repl, cen, ntr = self.stats_list()
        if flagged == 0:
            return
        self._last_report = now
        self.log(f"[flow_guard] call {self.calls}: {flagged}/{n} anchors flagged (no support after warp); "
                 f"{kept} kept (majority-flagged tracks: {ntr}, treated as fast objects), {repl} replaced by neighbours, "
                 f"{cen} set to the track centroid displacement")

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
    def __call__(self, result: FlowResult, current_points: torch.Tensor, current_track_ids: torch.Tensor | None = None) -> FlowResult:
        self.calls += 1
        anchors = result.source_anchors.float()
        flow = result.anchor_flow.float()
        track_ids = result.source_anchor_track_ids
        cur = current_points.float()
        support = _nearest_distance(anchors + flow, cur)
        # support radius from the data: the current cloud's own nearest-neighbour spacing (median), scaled
        radius = torch.tensor(self.support_radius_m, device=cur.device)
        if cur.shape[0] >= 8:
            sub = cur if cur.shape[0] <= 4096 else cur[torch.randperm(cur.shape[0], device=cur.device)[:4096]]
            from difflow3d.model.pointconv import knn_point
            nn2 = knn_point(2, sub.unsqueeze(0).contiguous(), sub.unsqueeze(0).contiguous())[0, :, 1].long()
            spacing = torch.linalg.vector_norm(sub - sub.index_select(0, nn2), dim=1).median()
            radius = torch.maximum(radius, self.support_spacing_factor * spacing)
        raw_flagged = (torch.linalg.vector_norm(flow, dim=1) > self.min_displacement_m) & (support > radius)
        # a track with a majority of its anchors flagged is not a mask-border bleed: keep its flow
        uniq, inv = torch.unique(track_ids, return_inverse=True)
        n_tr = torch.bincount(inv, minlength=uniq.shape[0]).float()
        f_tr = torch.bincount(inv, weights=raw_flagged.float(), minlength=uniq.shape[0])
        majority = (f_tr > self.majority_fraction * n_tr) & (n_tr >= 4)
        kept_mask = raw_flagged & majority[inv]
        flagged = raw_flagged & ~kept_mask
        self.last_flagged = flagged
        excluded_ids = torch.where(flagged, torch.full_like(track_ids, _EXCLUDED_TRACK), track_ids)

        new_flow, counts = self._interpolate(anchors, flow, track_ids, excluded_ids)
        has_neighbor = counts > 0
        # fallback without unflagged neighbours: the track's centroid displacement (never zero)
        fallback = flow
        if current_track_ids is not None and current_track_ids.shape[0] == cur.shape[0]:
            c_anchor = torch.zeros((uniq.shape[0], 3), device=cur.device).index_add_(0, inv, anchors) / n_tr[:, None].clamp_min(1.0)
            cur_inv = torch.searchsorted(uniq, current_track_ids)
            valid = (cur_inv < uniq.shape[0])
            cur_inv = cur_inv.clamp_max(uniq.shape[0] - 1)
            valid = valid & (uniq[cur_inv] == current_track_ids)
            n_cur = torch.bincount(cur_inv[valid], minlength=uniq.shape[0]).float()
            c_cur = torch.zeros((uniq.shape[0], 3), device=cur.device).index_add_(0, cur_inv[valid], cur[valid]) / n_cur[:, None].clamp_min(1.0)
            have = (n_cur > 0)[inv]
            fallback = torch.where(have[:, None], (c_cur - c_anchor)[inv], flow)
        replacement = torch.where(has_neighbor[:, None], new_flow, fallback)
        corrected_flow = torch.where(flagged[:, None], replacement, flow)
        self.last_stats = torch.stack([torch.tensor(anchors.shape[0], device=cur.device), raw_flagged.sum(), kept_mask.sum(),
                                       (flagged & has_neighbor).sum(), (flagged & ~has_neighbor).sum(), majority.sum()]).to(torch.int64)

        cov = result.anchor_flow_cov
        corrected_cov = cov
        if cov is not None:
            upper = pack_symmetric_upper(cov)
            parts = [self._interpolate(anchors, upper[:, k:k + 3], track_ids, excluded_ids)[0] for k in (0, 3)]
            new_cov = unpack_symmetric_upper(torch.cat(parts, dim=1))
            use_new = (flagged & has_neighbor)[:, None, None]
            corrected_cov = torch.where(use_new, new_cov.to(cov.dtype), cov)   # centroid fallback keeps its own covariance

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
