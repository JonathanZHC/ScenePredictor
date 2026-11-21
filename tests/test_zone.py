"""Zone of the pipeline: depth outside the box is blanked before anything is computed; the static memory still sees past it."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig, SceneCloudConfig, StaticMemoryConfig, ZoneConfig, load_config_from_mapping
from scene_pred_pipeline.data_types import CameraFrameCpu, MultiCameraFrame, SceneVelocityOutput
from scene_pred_pipeline.scene_assembly import SceneCloudAssembler
from scene_pred_pipeline.zone import ZoneFilter

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
DEV = torch.device("cuda")
H, W, F = 32, 48, 40.0


def _frame(depth, T=None):
    K = np.array([[F, 0.0, W / 2], [0.0, F, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32)
    cam = CameraFrameCpu(camera_name="camera_0", stamp_ns=0, rgb=np.zeros((H, W, 3), dtype=np.uint8), depth=depth.astype(np.float32), K=K,
                         T_world_camera=np.eye(4, dtype=np.float32) if T is None else T, optical_frame_id="camera_0_color_optical_frame")
    return MultiCameraFrame(stamp_ns=0, cameras={"camera_0": cam})


def _output():
    empty = torch.empty((0, 3), device=DEV); ids = torch.empty((0,), dtype=torch.int32, device=DEV)
    return SceneVelocityOutput(stamp_ns=0, flow_dt_s=0.033, tracked_points=empty, tracked_track_ids=ids, flow_points=empty, flow_velocity=empty,
                               flow_track_ids=ids, source_anchors=empty, warped_anchors=empty, removed_outlier_points=empty, view_results={},
                               common_track_ids=(), flow_valid=False, timings_ms={})


def test_config_default_is_no_zone_and_validation():
    assert PipelineConfig().zone is None
    base = {"scene_cloud": {"voxel_size_m": 0.02}}
    assert load_config_from_mapping(base, base_dir=".").zone is None
    assert load_config_from_mapping({**base, "zone": None}, base_dir=".").zone is None
    assert load_config_from_mapping({**base, "zone": "none"}, base_dir=".").zone is None
    z = load_config_from_mapping({**base, "zone": {"min": [-1, -1, 0], "max": [1, 1, 2]}}, base_dir=".").zone
    assert z.min == (-1.0, -1.0, 0.0) and z.max == (1.0, 1.0, 2.0) and z.pose == (0.0,) * 6
    for bad in ({"min": [0, 0, 0]}, {"min": [0, 0, 0], "max": [1, 0, 1]}, {"min": [0, 0, 0], "max": [1, 1, 1], "pose": [0, 0, 0]},
                {"min": [0, 0, 0], "max": [1, 1, 1], "box": 1}):
        with pytest.raises(ValueError):
            load_config_from_mapping({**base, "zone": bad}, base_dir=".")


def test_depth_outside_the_zone_is_blanked_and_the_original_kept():
    depth = np.full((H, W), 1.0, dtype=np.float32); depth[:, W // 2:] = 3.0; depth[0, 0] = np.nan
    zone = ZoneFilter(ZoneConfig(min=(-0.2, -5.0, 0.0), max=(5.0, 5.0, 2.0)), DEV)       # x >= -0.2, z <= 2
    frame = _frame(depth); zone.apply(frame)
    cam = frame.cameras["camera_0"]
    out = cam.depth_gpu.cpu().numpy()
    x = (np.arange(W) - W / 2) / F * 1.0
    expect = np.where((x >= -0.2)[None, :] & (depth == 1.0), 1.0, 0.0).astype(np.float32)
    assert np.array_equal(out, expect)                                                    # 3 m half and x < -0.2 gone, NaN -> 0
    assert np.array_equal(np.asarray(cam.depth), expect)                                  # host view follows
    assert np.array_equal(cam.depth_unzoned_gpu.cpu().numpy(), depth, equal_nan=True)     # what came in is still there
    ref = ZoneFilter(ZoneConfig(min=(-0.2, -5.0, 0.0), max=(5.0, 5.0, 2.0)), DEV); ref.use_kernel = False
    frame2 = _frame(depth); ref.apply(frame2)
    assert np.array_equal(frame2.cameras["camera_0"].depth_gpu.cpu().numpy(), expect)     # plain-torch path: the same


def test_zone_frame_pose_moves_and_turns_the_box():
    # zone frame = world shifted by 1 m in z and turned by 90 deg about z: zone x = world y, zone y = -world x
    cfg = ZoneConfig(min=(0.015, -5.0, -0.5), max=(5.0, 5.0, 0.5), pose=(0.0, 0.0, 1.0, 0.0, 0.0, np.pi / 2))
    depth = np.full((H, W), 1.2, dtype=np.float32)
    frame = _frame(depth); ZoneFilter(cfg, DEV).apply(frame)
    out = frame.cameras["camera_0"].depth_gpu.cpu().numpy()
    y_world = (np.arange(H) - H / 2) / F * 1.2
    assert np.array_equal(out > 0, np.broadcast_to((y_world >= 0.015)[:, None], (H, W)))  # zone x = world y; world z 1.2 lies in 0.5..1.5
    far = _frame(np.full((H, W), 1.6, dtype=np.float32)); ZoneFilter(cfg, DEV).apply(far)
    assert not bool(far.cameras["camera_0"].depth_gpu.any())                              # world z 1.6 = zone z 0.6 > 0.5


def test_memory_sees_past_the_zone():
    """An object inside the zone in front of a background outside it: taken away, it must be deleted by the rays that now
    reach the background - the visibility test reads the depth before the zone blanked it."""
    config = PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=0.02, depth_min_m=0.1, depth_max_m=5.0),
                            static_memory=StaticMemoryConfig(enabled=True, empty_frames_to_delete=2, unknown_frames_to_delete=0),
                            zone=ZoneConfig(min=(-5.0, -5.0, 0.0), max=(5.0, 5.0, 2.0)))
    asm = SceneCloudAssembler(config, tracker_config=None); zone = ZoneFilter(config.zone, DEV)

    def cycle(depth):
        frame = _frame(depth); zone.apply(frame); out = _output(); asm.assemble(out, frame); asm.finish_memory()
        return out.scene_points

    background = np.full((H, W), 3.0, dtype=np.float32)                # outside the zone (z <= 2)
    with_object = background.copy(); with_object[8:24, 12:36] = 1.0
    pts = cycle(with_object)
    assert pts.shape[0] > 0 and bool((pts[:, 2] < 2.0).all())          # the object only: nothing beyond the zone
    cycle(background); cycle(background)                               # two cycles of rays through it ...
    assert asm.static_memory.size == 0 and cycle(background).shape[0] == 0   # ... and it is gone, from the next cloud on


def test_lazy_host_depth_behaves_like_the_array_in_expressions():
    from scene_pred_pipeline.data_types import LazyHostDepth

    depth = LazyHostDepth(torch.tensor([[0.0, 1.5], [float("nan"), 0.4]]))
    valid = np.array([[True, True], [True, False]]) & np.isfinite(depth) & (depth > 0.0)     # as the tracker writes it
    assert valid.tolist() == [[False, True], [False, False]]
    assert ((depth >= 0.3) & (depth <= 1.0)).tolist() == [[False, False], [False, True]]
    assert np.allclose(2.0 * depth[0], [0.0, 3.0]) and float((depth * 2.0)[0, 1]) == 3.0
