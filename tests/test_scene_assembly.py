"""SceneCloudAssembler: merge order, zero velocity for rest points, workspace crop, rest backprojection."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig, SceneCloudConfig
from scene_pred_pipeline.data_types import CameraFrameCpu, MultiCameraFrame, SceneVelocityOutput
from scene_pred_pipeline.scene_assembly import SceneCloudAssembler

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = torch.device("cuda:0")


def _tracker_config(voxel=0.02):
    return SimpleNamespace(
        shared_voxel_grid=SimpleNamespace(voxel_size_m=voxel, origin_world=(0.0, 0.0, 0.0)),
        postprocess=SimpleNamespace(min_valid_depth_m=0.1, max_valid_depth_m=5.0),
    )


def _output(flow_valid=True, n_dyn=50, view_results=None):
    pts = torch.rand((n_dyn, 3), device=DEV) + torch.tensor([1.0, 0.0, 0.0], device=DEV)
    vel = torch.full((n_dyn, 3), 0.3, device=DEV)
    ids = torch.full((n_dyn,), 7, dtype=torch.int32, device=DEV)
    empty = torch.empty((0, 3), device=DEV)
    return SceneVelocityOutput(
        stamp_ns=0, flow_dt_s=0.033, tracked_points=pts, tracked_track_ids=ids,
        flow_points=pts if flow_valid else empty, flow_velocity=vel if flow_valid else empty,
        flow_track_ids=ids if flow_valid else torch.empty((0,), dtype=torch.int32, device=DEV),
        source_anchors=empty, warped_anchors=empty, removed_outlier_points=empty,
        view_results=view_results or {}, common_track_ids=(7,), flow_valid=flow_valid, timings_ms={},
    )


def _fake_view(height=8, width=8, depth_value=1.0):
    depth = np.full((height, width), depth_value, dtype=np.float32)
    frame = SimpleNamespace(
        depth_m=depth,
        intrinsics=SimpleNamespace(fx=10.0, fy=10.0, cx=width / 2, cy=height / 2),
        world_from_camera=np.eye(4, dtype=np.float32),
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[:, : width // 2] = True  # left half is a tracked object -> excluded from the rest scene
    return SimpleNamespace(frame=frame, instances=[SimpleNamespace(mask=mask)], exclusion_mask_gpu=None)


def test_dynamic_first_and_rest_zero_velocity():
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(enabled=True, include_rest_points=True), runtime=PipelineConfig().runtime)
    asm = SceneCloudAssembler(cfg, tracker_config=_tracker_config())
    out = _output(view_results={"cam": _fake_view()})
    asm.assemble(out)
    assert out.scene_num_dynamic == 50
    assert out.scene_points.shape[0] > 50
    assert torch.all(out.scene_velocity[:50] == 0.3)
    assert torch.all(out.scene_velocity[50:] == 0.0)
    assert torch.all(out.scene_track_ids[:50] == 7) and torch.all(out.scene_track_ids[50:] == 0)
    # 8x8 depth with the left half masked: at most 32 rest points, all at z = 1
    assert out.scene_points.shape[0] - 50 <= 32
    assert torch.allclose(out.scene_points[50:, 2], torch.ones(1, device=DEV))
    assert out.scene_points.is_contiguous() and out.scene_points.dtype == torch.float32


def test_flow_invalid_uses_tracked_points_with_zero_velocity():
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=False))
    asm = SceneCloudAssembler(cfg, tracker_config=_tracker_config())
    out = _output(flow_valid=False)
    asm.assemble(out)
    assert out.scene_num_dynamic == 50 and out.scene_points.shape[0] == 50
    assert torch.all(out.scene_velocity == 0.0)


def test_workspace_crop():
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=False, workspace_min=(1.0, 0.0, 0.0), workspace_max=(1.5, 1.0, 1.0)))
    asm = SceneCloudAssembler(cfg, tracker_config=_tracker_config())
    out = _output()
    asm.assemble(out)
    assert out.scene_points.shape[0] < 50
    assert torch.all(out.scene_points[:, 0] <= 1.5)
    assert out.scene_num_dynamic == out.scene_points.shape[0]


def _rgbd_frame(height=8, width=8, depth_value=1.0):
    K = np.array([[10.0, 0.0, width / 2], [0.0, 10.0, height / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    cam = CameraFrameCpu(
        camera_name="camera_0", stamp_ns=0, rgb=np.zeros((height, width, 3), dtype=np.uint8),
        depth=np.full((height, width), depth_value, dtype=np.float32), K=K,
        T_world_camera=np.eye(4, dtype=np.float32), optical_frame_id="camera_0_color_optical_frame",
    )
    return MultiCameraFrame(stamp_ns=0, cameras={"camera_0": cam})


def test_static_only_rest_points_from_frame_without_tracker():
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=True, voxel_size_m=0.02))
    asm = SceneCloudAssembler(cfg, tracker_config=None)          # no tracker at all
    out = _output(flow_valid=False, n_dyn=0)
    out.tracked_points = torch.empty((0, 3), device=DEV)
    out.tracked_track_ids = torch.empty((0,), dtype=torch.int32, device=DEV)
    asm.assemble(out, frame=_rgbd_frame())
    assert out.scene_num_dynamic == 0
    assert 0 < out.scene_points.shape[0] <= 64
    assert torch.all(out.scene_velocity == 0.0) and torch.all(out.scene_track_ids == 0)
    assert torch.allclose(out.scene_points[:, 2], torch.ones(1, device=DEV))


def test_voxel_size_override_downsamples():
    fine = PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=0.005))
    coarse = PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=0.5))
    n_fine = SceneCloudAssembler(fine).rest_points_from_frame(_rgbd_frame(32, 32)).shape[0]
    n_coarse = SceneCloudAssembler(coarse).rest_points_from_frame(_rgbd_frame(32, 32)).shape[0]
    assert n_fine > n_coarse >= 1


def test_tracker_lattice_is_default_but_override_wins():
    tc = _tracker_config(voxel=0.005)
    asm_default = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig()), tracker_config=tc)
    asm_override = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=0.03)), tracker_config=tc)
    assert asm_default.voxel_size_m == 0.005 and asm_override.voxel_size_m == 0.03


def test_scene_num_input_counts_raw_points_before_dedup():
    # 8x8 view, left half excluded -> 32 valid depth pixels enter the rest build; with a coarse
    # lattice they collapse to a handful of voxels, but scene_num_input reports the raw count.
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=True, voxel_size_m=0.5))
    asm = SceneCloudAssembler(cfg, tracker_config=_tracker_config(voxel=0.5))
    out = _output(n_dyn=50, view_results={"cam": _fake_view()})
    asm.assemble(out)
    assert out.scene_num_input == 50 + 32
    assert out.scene_points.shape[0] < out.scene_num_input
    assert out.scene_num_dynamic == 50


def test_reused_upload_buffers_match_a_fresh_assembler():
    # The depth/transform/mask uploads go through pinned buffers reused across cycles: a long-lived
    # assembler must give exactly what a fresh one gives for every frame of a changing sequence.
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=True, voxel_size_m=0.01))
    reused = SceneCloudAssembler(cfg, tracker_config=None)
    rng = np.random.default_rng(3)
    for step in range(6):
        frame = _rgbd_frame(24, 32)
        cam = frame.cameras["camera_0"]
        cam.depth[...] = rng.uniform(0.5, 3.0, cam.depth.shape).astype(np.float32)
        cam.T_world_camera[:3, 3] = rng.uniform(-1.0, 1.0, 3).astype(np.float32)
        got = reused.rest_points_from_frame(frame)
        want = SceneCloudAssembler(cfg, tracker_config=None).rest_points_from_frame(frame)
        assert torch.equal(got, want), f"cycle {step}"
        assert int(reused.last_rest_input) == int(np.isfinite(cam.depth).sum())


def test_fused_rest_scene_matches_float64_reference():
    # The fused Triton backprojection (rest_scene_kernel.py): fp32 world points within 1e-5 m of a float64 reference
    # (the former GEMM ran in TF32: up to 1.4 mm off), voxel dedup on the same lattice, excluded pixels dropped.
    from scene_pred_pipeline import rest_scene_kernel

    if not rest_scene_kernel.available(DEV):
        pytest.skip("Triton not available")
    cfg = PipelineConfig(scene_cloud=SceneCloudConfig(include_rest_points=True, voxel_size_m=0.005))
    asm = SceneCloudAssembler(cfg, tracker_config=None)
    rng = np.random.default_rng(5)
    h, w = 40, 60
    depth = rng.uniform(0.3, 3.0, (h, w)).astype(np.float32)
    depth[0, :5] = np.nan
    depth[1, :5] = 0.05                     # below min depth
    angle = 0.7
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], np.float32)
    T[:3, 3] = (0.4, -1.2, 0.8)
    excluded = torch.zeros((h, w), dtype=torch.bool, device=DEV)
    excluded[20:30, 30:50] = True
    views = [(depth, (80.0, 80.0, w / 2, h / 2), T, excluded)]
    fused = asm._rest_points_from_views(views)
    # float64 reference of every valid, non-excluded pixel
    v, u = np.mgrid[0:h, 0:w]
    z = depth.astype(np.float64)
    ok = np.isfinite(z) & (z >= asm._min_depth_m) & (z <= asm._max_depth_m) & ~excluded.cpu().numpy()
    pc = np.stack([(u - w / 2) / 80.0 * z, (v - h / 2) / 80.0 * z, z], -1)[ok]
    ref = pc @ T[:3, :3].astype(np.float64).T + T[:3, 3]
    keys = {tuple(k) for k in np.floor(ref / 0.005).astype(np.int64)}
    got = fused.cpu().numpy().astype(np.float64)
    assert int(asm.last_rest_input) == int(ok.sum())
    assert abs(len(got) - len(keys)) <= 2          # voxel boundaries may round differently in fp32
    dist = np.abs(got[:, None, :] - ref[None, :, :]).max(-1).min(1)      # <= 2400 x 2400
    assert dist.max() < 1e-5
