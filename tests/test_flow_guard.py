"""Segmentation-bleed guard: flagging, neighbour-softmax replacement of displacement and covariance, config parsing."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scene_pred_pipeline.config import DifFlowFlowGuardConfig, _load_difflow_config as load_difflow_config
from scene_pred_pipeline.data_types import FlowResult
from scene_pred_pipeline.flow_guard import FlowGuard

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = torch.device("cuda:0")
SIGMA = 0.025


class _TrackAwareRecoverer:
    """Torch reference of the local track-aware Gaussian-softmax recovery: anchors of another track or outside
    4 sigma are ignored; local_neighbor_counts = number of usable anchors (0 -> the kernel's global fallback)."""

    def recover(self, *, query_points, anchor_points, anchor_flow, dt_s, query_track_ids, anchor_track_ids):
        d2 = torch.cdist(query_points.float(), anchor_points.float()) ** 2
        ok = (query_track_ids[:, None] == anchor_track_ids[None, :]) & (d2 <= (4 * SIGMA) ** 2)
        logits = torch.where(ok, -d2 / (2 * SIGMA ** 2), torch.full_like(d2, -1e30))
        w = torch.softmax(logits, dim=1) * ok
        flow = w @ anchor_flow.float() / w.sum(1, keepdim=True).clamp_min(1e-30)
        return SimpleNamespace(flow=flow, local_neighbor_counts=ok.sum(1).int())


def _spd(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((n, 3, 3), generator=g)
    return (a @ a.transpose(1, 2) * 1e-5 + 1e-7 * torch.eye(3)).to(DEV)


def _scene():
    """A 10 x 10 grid of anchors (1 cm apart) moving +2 cm in x, one anchor with a spurious 40 cm jump, and a
    second, isolated anchor (30 cm away) with a spurious jump and no neighbours."""
    xs = torch.stack(torch.meshgrid(torch.arange(10.0), torch.arange(10.0), indexing="ij"), -1).reshape(-1, 2) * 0.01
    grid = torch.cat([xs, torch.ones(100, 1)], 1)
    isolated = torch.tensor([[0.40, 0.40, 1.0]])
    anchors = torch.cat([grid, isolated]).to(DEV)
    flow = torch.zeros_like(anchors); flow[:, 0] = 0.02
    bad = 55
    flow[bad] = torch.tensor([0.0, 0.0, 0.40], device=DEV)
    flow[100] = torch.tensor([0.0, -0.30, 0.0], device=DEV)
    current = torch.cat([grid.to(DEV)[:100] + torch.tensor([0.02, 0.0, 0.0], device=DEV), isolated.to(DEV)])   # the isolated point stays
    cov = _spd(101)
    result = FlowResult(source_anchors=anchors, warped_anchors=anchors + flow, anchor_flow=flow,
                        source_anchor_track_ids=torch.ones(101, dtype=torch.int32, device=DEV),
                        target_input_keep_mask=torch.ones(101, dtype=torch.bool, device=DEV),
                        anchor_flow_cov=cov, anchor_flow_cov_scale=torch.full((101,), 2.5, device=DEV))
    return result, current, bad


def _guard(scale=None):
    return FlowGuard(_TrackAwareRecoverer(), min_displacement_m=0.05, support_radius_m=0.015, calibration_scale=scale)


def test_flags_only_unsupported_large_displacements():
    result, current, bad = _scene()
    guard = _guard()
    guard(result, current)
    flagged = guard.last_flagged.cpu()
    assert flagged[bad] and flagged[100] and int(flagged.sum()) == 2


def test_replacement_is_neighbour_softmax_and_others_untouched():
    result, current, bad = _scene()
    out = _guard()(result, current)
    keep = torch.ones(101, dtype=torch.bool); keep[bad] = False; keep[100] = False
    torch.testing.assert_close(out.anchor_flow[keep.to(DEV)], result.anchor_flow[keep.to(DEV)])
    torch.testing.assert_close(out.anchor_flow_cov[keep.to(DEV)], result.anchor_flow_cov[keep.to(DEV)])
    # the bad anchor sits among neighbours moving +2 cm in x: it gets exactly that
    torch.testing.assert_close(out.anchor_flow[bad], torch.tensor([0.02, 0.0, 0.0], device=DEV), atol=1e-6, rtol=0)
    torch.testing.assert_close(out.warped_anchors[bad], result.source_anchors[bad] + out.anchor_flow[bad])
    # its covariance is the softmax mixture of the unflagged neighbours' covariances (excluding itself)
    ref = _TrackAwareRecoverer()
    ids = result.source_anchor_track_ids.clone(); ids[bad] = -7; ids[100] = -7
    upper = torch.stack([result.anchor_flow_cov[:, i, j] for i, j in ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))], 1)
    w = torch.cat([ref.recover(query_points=result.source_anchors, anchor_points=result.source_anchors, anchor_flow=upper[:, k:k + 3],
                               dt_s=1.0, query_track_ids=result.source_anchor_track_ids, anchor_track_ids=ids).flow for k in (0, 3)], 1)
    assert torch.allclose(out.anchor_flow_cov[bad, 0, 0], w[bad, 0]) and torch.allclose(out.anchor_flow_cov[bad, 1, 0], w[bad, 3])
    assert torch.allclose(out.anchor_flow_cov[bad], out.anchor_flow_cov[bad].T)
    assert bool((torch.linalg.eigvalsh(out.anchor_flow_cov[bad].double()) > 0).all())


def test_isolated_flagged_anchor_falls_back_to_the_track_centroid_displacement_never_zero():
    result, current, bad = _scene()
    out = _guard()(result, current)                                       # no current track ids: keeps its own displacement
    torch.testing.assert_close(out.anchor_flow[100], result.anchor_flow[100])
    torch.testing.assert_close(out.anchor_flow_cov[100], result.anchor_flow_cov[100])
    ids = torch.ones(current.shape[0], dtype=torch.int32, device=DEV)
    out = _guard()(result, current, ids)                                  # track centroid moved +2 cm x (100 of 101 points)
    torch.testing.assert_close(out.anchor_flow[100], torch.tensor([0.02 * 100 / 101, 0.0, 0.0], device=DEV), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.anchor_flow_cov[100], result.anchor_flow_cov[100])
    assert out.anchor_flow[100].abs().sum() > 0


def test_majority_flagged_track_keeps_its_flow():
    """A fast object: all anchors of a track jump 20 cm and none has support -> not bleed, flow kept, reported."""
    result, current, bad = _scene()
    anchors2 = result.source_anchors + torch.tensor([2.0, 0.0, 0.0], device=DEV)
    flow2 = torch.zeros_like(anchors2); flow2[:, 0] = 0.20
    ids2 = torch.full((101,), 7, dtype=torch.int32, device=DEV)
    merged = FlowResult(source_anchors=torch.cat([result.source_anchors, anchors2]), warped_anchors=torch.cat([result.warped_anchors, anchors2 + flow2]),
                        anchor_flow=torch.cat([result.anchor_flow, flow2]), source_anchor_track_ids=torch.cat([result.source_anchor_track_ids, ids2]),
                        target_input_keep_mask=torch.ones(202, dtype=torch.bool, device=DEV), anchor_flow_cov=None, anchor_flow_cov_scale=None)
    out = _guard()(merged, current)                                       # no current points of track 7 at all
    torch.testing.assert_close(out.anchor_flow[101:], flow2)
    n, flagged, kept, repl, cen, ntr = _guard_stats(_guard(), merged, current)
    assert n == 202 and flagged >= 101 + 2 and kept == 101 and ntr == 1


def _guard_stats(guard, result, current):
    guard(result, current)
    return guard.stats_list()


def test_support_radius_follows_the_cloud_spacing():
    """Current points 2 cm apart: support radius = 2.5 x 2 cm = 5 cm, so a 4 cm gap after warping is still supported."""
    result, current, bad = _scene()
    sparse = current[::2]                                                 # 2 cm spacing along one axis
    anchors = result.source_anchors[:50]
    flow = torch.zeros_like(anchors); flow[:, 0] = 0.06                   # big displacement, lands 4 cm off the sparse cloud
    r = FlowResult(source_anchors=anchors, warped_anchors=anchors + flow, anchor_flow=flow, source_anchor_track_ids=torch.ones(50, dtype=torch.int32, device=DEV),
                   target_input_keep_mask=torch.ones(50, dtype=torch.bool, device=DEV), anchor_flow_cov=None, anchor_flow_cov_scale=None)
    g = _guard(); g(r, sparse)
    assert g.stats_list()[1] == 0


def test_calibration_scale_follows_corrected_displacement():
    result, current, bad = _scene()
    out = _guard(scale=lambda f: torch.linalg.vector_norm(f, dim=1) * 100.0)(result, current)
    assert float(out.anchor_flow_cov_scale[bad]) == pytest.approx(2.0, rel=1e-4)       # |0.02 m| x 100
    assert float(out.anchor_flow_cov_scale[0]) == pytest.approx(2.5)                     # unflagged: unchanged


def test_without_covariance():
    result, current, bad = _scene()
    result = FlowResult(**{**result.__dict__, "anchor_flow_cov": None, "anchor_flow_cov_scale": None})
    out = _guard()(result, current)
    assert out.anchor_flow_cov is None and out.anchor_flow_cov_scale is None
    torch.testing.assert_close(out.anchor_flow[bad], torch.tensor([0.02, 0.0, 0.0], device=DEV), atol=1e-6, rtol=0)


def test_real_recoverer_matches_reference():
    pytest.importorskip("difflow3d.runtime")
    from difflow3d.runtime import SoftmaxAnchorMotionRecoverer
    try:
        real = SoftmaxAnchorMotionRecoverer(chunk_size=4096, softmax_sigma_m=SIGMA, backend="local", local_radius_sigma=4.0,
                                            local_hash_size_factor=4.0)
    except Exception as error:  # pragma: no cover - the CUDA extension is missing
        pytest.skip(f"local recovery backend unavailable: {error}")
    result, current, bad = _scene()
    out = FlowGuard(real, min_displacement_m=0.05, support_radius_m=0.015)(result, current)
    torch.testing.assert_close(out.anchor_flow[bad], torch.tensor([0.02, 0.0, 0.0], device=DEV), atol=1e-5, rtol=0)
    torch.testing.assert_close(out.anchor_flow[100], result.anchor_flow[100])     # isolated, no track info: kept, not zeroed


def test_config_defaults_and_validation(tmp_path):
    assert DifFlowFlowGuardConfig() == DifFlowFlowGuardConfig(enabled=False, min_displacement_m=0.05, support_radius_m=0.015,
                                                             support_spacing_factor=2.5, majority_fraction=0.5, report_period_s=1.0)
    base = "model:\n  checkpoint: x.pth\n"
    for text, message in (("flow_guard:\n  enabled: 1\n", "enabled must be boolean"),
                          ("flow_guard:\n  support_radius_m: -0.01\n", "must be positive"),
                          ("flow_guard:\n  radius: 0.01\n", "Unknown keys")):
        path = tmp_path / "d.yaml"; path.write_text(base + text)
        with pytest.raises(ValueError, match=message):
            load_difflow_config(path)
    path = tmp_path / "ok.yaml"; path.write_text(base + "flow_guard:\n  enabled: true\n  min_displacement_m: 0.06\n")
    cfg = load_difflow_config(path).flow_guard
    assert cfg.enabled is True and cfg.min_displacement_m == 0.06 and cfg.support_radius_m == 0.015
