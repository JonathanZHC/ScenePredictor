"""Velocity regions for the safety filter's tightening: anchor regions, recovery to the flow points, scene assembly."""
from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig, SceneCloudConfig
from scene_pred_pipeline.data_types import SceneVelocityOutput
from scene_pred_pipeline.scene_assembly import SceneCloudAssembler
from scene_pred_pipeline.velocity_recovery import VelocityRecovery, anchor_velocity_region

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = torch.device("cuda:0")


def _spd(count, seed=0):
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((count, 3, 3), generator=g, dtype=torch.float64)
    return (a @ a.transpose(1, 2) * 1e-5 + 1e-7 * torch.eye(3, dtype=torch.float64)).float().to(DEV)


def test_anchor_region_ellipsoid_and_ball():
    cov, scale, dt = _spd(5), torch.tensor([1.0, 2.0, 3.0, 2.5, 5.6], device=DEV), 0.033
    ellipsoid = anchor_velocity_region(cov, scale, dt, "ellipsoid")
    torch.testing.assert_close(ellipsoid, cov * (scale / dt)[:, None, None] ** 2)
    ball = anchor_velocity_region(cov, scale, dt, "ball")
    radius = scale * torch.linalg.det(cov.double()).float() ** (1.0 / 6.0) / dt
    torch.testing.assert_close(ball, radius[:, None, None] ** 2 * torch.eye(3, device=DEV), rtol=1e-4, atol=1e-6)


class _Recoverer:
    """Stub of SoftmaxAnchorMotionRecoverer: query q takes anchor index[q]; records the call kwargs."""

    backend = "local"

    def __init__(self, index):
        self.index, self.calls = index, []

    def recover(self, *, query_points, anchor_points, anchor_flow, dt_s, **kwargs):
        assert anchor_flow.shape == (anchor_points.shape[0], 3) and dt_s == 1.0
        self.calls.append(kwargs)
        return SimpleNamespace(flow=anchor_flow[self.index])


def _recovery(index, same_track):
    recovery = object.__new__(VelocityRecovery)
    recovery.config = SimpleNamespace(recovery=SimpleNamespace(restrict_same_track=same_track))
    recovery.recoverer = _Recoverer(index)
    return recovery


@pytest.mark.parametrize("same_track", [False, True])
def test_recover_region_rebuilds_symmetric_matrices(same_track):
    anchors, region = torch.rand((6, 3), device=DEV), _spd(6, seed=1)
    index = torch.tensor([3, 0, 5, 5, 1], device=DEV)
    flow_result = SimpleNamespace(warped_anchors=anchors, source_anchor_track_ids=torch.ones(6, dtype=torch.int32, device=DEV))
    recovery = _recovery(index, same_track)
    query_ids = torch.ones(5, dtype=torch.int32, device=DEV)
    out = recovery.recover_region(flow_result, region, query_points=torch.rand((5, 3), device=DEV), query_track_ids=query_ids)
    torch.testing.assert_close(out, region[index])
    torch.testing.assert_close(out, out.transpose(1, 2))
    assert len(recovery.recoverer.calls) == 2                       # six unique entries in two 3-channel passes
    assert all(("query_track_ids" in kwargs) == same_track for kwargs in recovery.recoverer.calls)
    empty = recovery.recover_region(flow_result, region, query_points=torch.empty((0, 3), device=DEV), query_track_ids=query_ids[:0])
    assert empty.shape == (0, 3, 3)


def _output(n_dyn=40, region=True, extra=0):
    pts = torch.rand((n_dyn, 3), device=DEV) + torch.tensor([1.0, 0.0, 0.0], device=DEV)
    vel = torch.full((n_dyn, 3), 0.3, device=DEV)
    ids = torch.full((n_dyn,), 7, dtype=torch.int32, device=DEV)
    tracked, tracked_ids = pts, ids
    if extra:                                                        # a tracked id without flow this cycle
        tracked = torch.cat((pts, torch.rand((extra, 3), device=DEV) + 1.0))
        tracked_ids = torch.cat((ids, torch.full((extra,), 9, dtype=torch.int32, device=DEV)))
    empty = torch.empty((0, 3), device=DEV)
    return SceneVelocityOutput(
        stamp_ns=0, flow_dt_s=0.033, tracked_points=tracked, tracked_track_ids=tracked_ids,
        flow_points=pts, flow_velocity=vel, flow_track_ids=ids,
        source_anchors=empty, warped_anchors=empty, removed_outlier_points=empty,
        view_results={}, common_track_ids=(7,), flow_valid=True, timings_ms={},
        flow_velocity_region=_spd(n_dyn, seed=2) if region else None,
    )


def test_scene_region_follows_the_dynamic_rows():
    asm = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=False)))
    out = _output(extra=3)
    asm.assemble(out)
    region = out.scene_velocity_region
    assert region.shape == (out.scene_num_dynamic, 3, 3) == (43, 3, 3)
    torch.testing.assert_close(region[:40], out.flow_velocity_region)
    assert torch.count_nonzero(region[40:]) == 0                     # tracked without flow: no region, no tightening


def test_scene_region_is_cropped_with_its_rows():
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=False, workspace_min=(1.0, 0.0, 0.0),
                                                      workspace_max=(1.5, 1.0, 1.0)))
    out = _output()
    keep = ((out.flow_points >= torch.tensor([1.0, 0.0, 0.0], device=DEV))
            & (out.flow_points <= torch.tensor([1.5, 1.0, 1.0], device=DEV))).all(dim=1)
    SceneCloudAssembler(cfg).assemble(out)
    assert out.scene_velocity_region.shape[0] == out.scene_num_dynamic == int(keep.sum())
    torch.testing.assert_close(out.scene_velocity_region, out.flow_velocity_region[keep])


def test_no_region_without_flow_regions():
    out = _output(region=False)
    SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=False))).assemble(out)
    assert out.scene_velocity_region is None
