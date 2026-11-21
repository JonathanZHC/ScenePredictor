"""StaticMemory: rest points stay while hidden, go when a ray sees through their place (synthetic depth images)."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig, SceneCloudConfig, StaticMemoryConfig, load_config_from_mapping
from scene_pred_pipeline.data_types import CameraFrameCpu, MultiCameraFrame, SceneVelocityOutput
from scene_pred_pipeline.scene_assembly import SceneCloudAssembler
from scene_pred_pipeline.static_memory import StaticMemory

if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

DEV = torch.device("cuda:0")
H = W = 32
FX = FY = 32.0
VOXEL = 0.02


def _K():
    return np.array([[FX, 0.0, W / 2], [0.0, FY, H / 2], [0.0, 0.0, 1.0]], dtype=np.float32)


def _frame(depth: np.ndarray, T=None, name="camera_0", stamp=0, mask=None):
    cam = CameraFrameCpu(camera_name=name, stamp_ns=stamp, rgb=np.zeros((H, W, 3), dtype=np.uint8),
                         depth=depth.astype(np.float32), K=_K(),
                         T_world_camera=np.eye(4, dtype=np.float32) if T is None else T,
                         optical_frame_id=f"{name}_color_optical_frame", occluder_mask=mask)
    return MultiCameraFrame(stamp_ns=stamp, cameras={name: cam})


def _assembler(**memory):
    memory.setdefault("min_observations_to_hold", 1)       # held from the first observation unless a test says otherwise
    config = PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=VOXEL, depth_min_m=0.1, depth_max_m=5.0),
                            static_memory=StaticMemoryConfig(enabled=True, **memory))
    return SceneCloudAssembler(config, tracker_config=None)


def _output():
    """Static-only cycle output: no tracked instances, no flow."""
    empty = torch.empty((0, 3), device=DEV)
    ids = torch.empty((0,), dtype=torch.int32, device=DEV)
    return SceneVelocityOutput(
        stamp_ns=0, flow_dt_s=0.033, tracked_points=empty, tracked_track_ids=ids, flow_points=empty,
        flow_velocity=empty, flow_track_ids=ids, source_anchors=empty, warped_anchors=empty,
        removed_outlier_points=empty, view_results={}, common_track_ids=(), flow_valid=False, timings_ms={},
    )


def _cycle(asm, depth, T=None, mask=None, forget=None, near=None):
    """One cycle: the cloud, then the deferred half of the memory (forget / near: functions cloud -> rest rows)."""
    out = _output()
    asm.assemble(out, _frame(depth, T, mask=mask))
    pts = out.scene_points
    asm.finish_memory(forget(pts) if forget else None, near(pts) if near else None)
    return pts


def test_config_default_off_and_validation():
    cfg = load_config_from_mapping({"tracker": {"tracked_prompts": []}, "scene_cloud": {"voxel_size_m": 0.005}}, base_dir=".")
    assert cfg.static_memory.enabled is False
    cfg = load_config_from_mapping({"tracker": {"tracked_prompts": []}, "scene_cloud": {"voxel_size_m": 0.005},
                                    "static_memory": {"enabled": True, "empty_frames_to_delete": 3}}, base_dir=".")
    assert cfg.static_memory.enabled and cfg.static_memory.empty_frames_to_delete == 3
    with pytest.raises(ValueError):
        load_config_from_mapping({"tracker": {"tracked_prompts": []}, "scene_cloud": {"voxel_size_m": 0.005},
                                  "static_memory": {"neighbourhood": 2}}, base_dir=".")


def test_memory_off_leaves_assembler_unchanged():
    config = PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=VOXEL))
    asm = SceneCloudAssembler(config, tracker_config=None)
    assert asm.static_memory is None
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n1 = _cycle(asm, wall).shape[0]
    n2 = _cycle(asm, wall).shape[0]
    assert n1 == n2 > 0


def test_same_view_does_not_grow_the_memory():
    asm = _assembler()
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n1 = _cycle(asm, wall).shape[0]
    for _ in range(3):
        assert _cycle(asm, wall).shape[0] == n1
    assert asm.static_memory.size == n1


def test_occluded_points_are_kept():
    asm = _assembler()
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n_wall = _cycle(asm, wall).shape[0]
    occluded = wall.copy()
    occluded[:, : W // 2] = 0.5                      # a plate in front of the left half of the wall
    n_occ = _cycle(asm, occluded).shape[0]
    plate_only = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=VOXEL)), tracker_config=None)
    n_plate_scene = _cycle(plate_only, occluded).shape[0]        # plate + visible right half of the wall
    assert n_occ > n_plate_scene                                  # the hidden left half of the wall is still there
    assert n_occ > n_wall                                         # = wall (kept) + plate (new)
    for _ in range(5):                                           # and stays while the plate stays
        assert _cycle(asm, occluded).shape[0] == n_occ


def test_robot_pixels_without_depth_keep_the_points():
    asm = _assembler(unknown_frames_to_delete=3)
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n_wall = _cycle(asm, wall).shape[0]
    masked = wall.copy()
    masked[8:24, 8:24] = 0.0                          # self-filter removed the robot: no depth there
    robot = np.zeros((H, W), dtype=bool); robot[8:24, 8:24] = True
    for _ in range(8):                                # far beyond the unknown timeout: robot pixels keep the points
        assert _cycle(asm, masked, mask=robot).shape[0] == n_wall
    asm2 = _assembler(unknown_frames_to_delete=3)
    _cycle(asm2, wall)
    for _ in range(2):                                # a CUDA mask works the same way
        assert _cycle(asm2, masked, mask=torch.as_tensor(robot, device=DEV)).shape[0] == n_wall


def test_points_without_any_evidence_expire():
    """A hole in the depth image that is not a robot pixel keeps the points only unknown_frames_to_delete cycles."""
    asm = _assembler(unknown_frames_to_delete=3)
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n_wall = _cycle(asm, wall).shape[0]
    holed = wall.copy()
    holed[8:24, 8:24] = 0.0                           # no depth, no mask: unknown
    n_seen = _cycle(asm, holed).shape[0]              # cycle 1: unknown_count 1
    assert n_seen == n_wall
    assert _cycle(asm, holed).shape[0] == n_wall      # 2
    assert _cycle(asm, holed).shape[0] == n_wall      # 3: dropped by this cycle's update, which runs after the cloud is out
    n3 = _cycle(asm, holed).shape[0]                  # ... so the cloud shows it one cycle later
    assert n3 < n_wall
    assert n3 == _cycle(asm, holed).shape[0]          # stable afterwards
    assert _cycle(asm, wall).shape[0] == n_wall       # the surface comes back when it is seen again
    # occlusion resets the timer: something in front keeps the points without limit
    asm = _assembler(unknown_frames_to_delete=2)
    _cycle(asm, wall)
    front = wall.copy(); front[8:24, 8:24] = 0.5
    for _ in range(6):
        assert _cycle(asm, front).shape[0] >= n_wall  # wall kept (plus the new object's points)


def test_empty_rays_delete_after_confirmation():
    asm = _assembler(empty_frames_to_delete=2)
    near = np.full((H, W), 1.0, dtype=np.float32)
    far = np.full((H, W), 2.0, dtype=np.float32)
    n_near = _cycle(asm, near).shape[0]
    n_far_alone = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=VOXEL)), tracker_config=None)
    n_far = _cycle(n_far_alone, far).shape[0]
    first = _cycle(asm, far).shape[0]                 # judged empty once: still remembered
    assert first == n_near + n_far
    second = _cycle(asm, far).shape[0]                # confirmed by the update of this cycle, after its cloud went out
    assert second == n_near + n_far and asm.static_memory.size == n_far
    assert _cycle(asm, far).shape[0] == n_far         # the old wall is gone from the cloud


def test_partial_removal_only_where_rays_pass():
    asm = _assembler(empty_frames_to_delete=1)
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n_wall = _cycle(asm, wall).shape[0]
    moved = wall.copy()
    moved[:, : W // 2] = 2.0                          # left half taken away, a far wall appears behind it
    n_after = _cycle(asm, moved).shape[0]
    mem = asm.static_memory
    left = mem.points[mem.points[:, 2] < 1.5]
    # No remembered near-wall point left of the optical axis, except the one pixel column at the boundary: its 3x3
    # neighbourhood still contains near-wall pixels, so the (conservative) empty test does not fire there.
    one_pixel = 1.0 / FX
    assert bool((left[:, 0] >= -one_pixel - 1e-3).all())
    assert n_after < n_wall + n_wall // 2 + 10


def test_second_camera_evidence_counts():
    # Camera A looks at a wall; camera B, placed 0.5 m to the side, sees past the wall's position from frame 2 on.
    asm = _assembler(empty_frames_to_delete=1)
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n_wall = _cycle(asm, wall).shape[0]
    out = _output()
    T_b = np.eye(4, dtype=np.float32)
    T_b[0, 3] = 0.5                                   # same orientation, shifted in x
    occluded_a = np.zeros((H, W), dtype=np.float32)   # camera A: robot in front, no depth at all
    far_b = np.full((H, W), 3.0, dtype=np.float32)    # camera B: nothing until 3 m -> the wall is gone
    frame = MultiCameraFrame(stamp_ns=1, cameras={
        "camera_0": _frame(occluded_a).cameras["camera_0"],
        "camera_1": _frame(far_b, T_b, name="camera_1").cameras["camera_1"],
    })
    asm.assemble(out, frame); asm.finish_memory()
    mem = asm.static_memory
    near = mem.points[(mem.points[:, 2] - 1.0).abs() < 0.05]
    # the part of the old wall inside camera B's view (x in roughly [0, 1] m at z = 1) was carved by B
    assert near.shape[0] < n_wall
    assert bool((near[:, 0] < 0.03).all())


def test_noise_voxels_beside_a_seen_surface_are_dropped():
    asm = _assembler(seen_unrefreshed_frames_to_delete=3)
    rng = np.random.default_rng(0)
    noisy = (1.0 + 0.004 * rng.standard_normal((H, W))).astype(np.float32)   # one frame of depth noise
    n_noisy = _cycle(asm, noisy).shape[0]
    clean = np.full((H, W), 1.0, dtype=np.float32)
    n_clean = SceneCloudAssembler(PipelineConfig(scene_cloud=SceneCloudConfig(voxel_size_m=VOXEL)), tracker_config=None)
    n_clean = _cycle(n_clean, clean).shape[0]
    for _ in range(2):
        assert _cycle(asm, clean).shape[0] >= n_noisy          # seen, not yet stale: still kept
    for _ in range(4):
        n = _cycle(asm, clean).shape[0]
    assert n == n_clean                                      # the noise shell is gone, the surface stays


def test_fused_kernel_matches_reference_path():
    from scene_pred_pipeline import static_memory_kernel
    if not static_memory_kernel.available(DEV):
        pytest.skip("Triton not available")
    rng = np.random.default_rng(1)
    frames = []
    for k in range(6):
        d = (1.0 + 0.003 * rng.standard_normal((H, W))).astype(np.float32)
        d[8:20, 4 + k:16 + k] = 0.5 if k % 2 else 0.0                    # a plate in front / robot pixels, moving
        if k >= 3:
            d[:, W // 2:] = 2.0                                           # right half taken away
        frames.append(d)
    fused = _assembler(empty_frames_to_delete=1, seen_unrefreshed_frames_to_delete=2)
    ref = _assembler(empty_frames_to_delete=1, seen_unrefreshed_frames_to_delete=2, reference_path=True)
    assert fused.static_memory.use_kernel and not ref.static_memory.use_kernel
    for d in frames:
        a = _cycle(fused, d); b = _cycle(ref, d)
        assert a.shape == b.shape
        assert torch.equal(fused.static_memory.keys, ref.static_memory.keys)


def test_points_behind_the_edge_of_the_robot_are_not_judged_empty():
    """A remembered surface whose pixel is a robot pixel stays, also at the robot's silhouette where the neighbouring
    pixels see far beyond it (those rays pass beside the point, not through it)."""
    for reference in (False, True):
        asm = _assembler(reference_path=reference)
        scene = np.full((H, W), 1.0, dtype=np.float32)
        scene[8:24, 8:24] = 0.5                       # a small plate in front of the wall
        n = _cycle(asm, scene).shape[0]
        hidden = scene.copy(); hidden[8:24, 8:24] = 0.0
        robot = np.zeros((H, W), dtype=bool); robot[8:24, 8:24] = True     # the robot now covers exactly the plate
        for _ in range(6):
            assert _cycle(asm, hidden, mask=robot).shape[0] == n


def test_young_points_beside_the_robot_are_not_held():
    """What shows up for a frame or two beside the robot (a robot pixel the self-filter missed) and is then hidden by the
    robot is dropped; the same thing observed for a few cycles, or anywhere the consumer did not report as near, is held."""
    robot = np.zeros((H, W), dtype=bool); robot[8:24, 8:24] = True
    wall = np.full((H, W), 1.0, dtype=np.float32)
    blob = wall.copy(); blob[8:24, 8:24] = 0.5                         # something at 0.5 m in front of the wall
    hidden = wall.copy(); hidden[8:24, 8:24] = 0.0                     # ... and then robot pixels there
    for cycles_seen, report_near, held in ((1, True, False), (2, True, False), (3, True, True), (6, True, True), (1, False, True)):
        asm = _assembler(min_observations_to_hold=3, robot_born_needs_anchor=False)     # the age rule alone
        mem = asm.static_memory

        def cycle(depth, mask=None):                                   # the consumer reports the rows beside the robot
            pts = _cycle(asm, depth, mask=mask, near=(lambda p: (p[:, 2] < 0.75).nonzero(as_tuple=True)[0]) if report_near else None)
            return int((pts[:, 2] < 0.75).sum())

        for _ in range(cycles_seen):
            n_blob = cycle(blob)
        assert n_blob > 0
        for _ in range(4):
            near = cycle(hidden, mask=robot)
        assert near == (n_blob if held else 0), (cycles_seen, report_near, near, n_blob)


def test_robot_born_points_need_an_anchor():
    """Born at the robot's silhouette and connected to nothing else: never held, however often it was observed. Born there
    but part of a surface that reaches beyond the silhouette: held like any other point."""
    robot = np.zeros((H, W), dtype=bool); robot[8:24, 4:10] = True               # the arm, a vertical bar in the image
    all_robot = np.ones((H, W), dtype=bool); nothing = np.zeros((H, W), dtype=np.float32)
    # a speck right beside the arm, in front of a far background that never enters the cloud (beyond the depth range)
    asm = _assembler(silhouette_px=4)
    speck = np.full((H, W), 9.0, dtype=np.float32); speck[8:24, 4:10] = 0.0; speck[12:16, 10:13] = 0.6
    for _ in range(8):
        n_speck = _cycle(asm, speck, mask=robot).shape[0]
    assert n_speck > 0 and asm.static_memory.size == n_speck                     # observed: in the cloud and in the memory
    assert _cycle(asm, nothing, mask=all_robot).shape[0] == 0                    # hidden by the arm: not held
    assert asm.static_memory.size == 0
    # the same with the rule off: held
    off = _assembler(silhouette_px=4, robot_born_needs_anchor=False)
    for _ in range(8):
        _cycle(off, speck, mask=robot)
    assert _cycle(off, nothing, mask=all_robot).shape[0] == n_speck
    # a wall that the arm uncovers: its voxels beside the arm are robot-born, the rest of the wall anchors them
    asm = _assembler(silhouette_px=4)
    wall = np.full((H, W), 0.5, dtype=np.float32); wall[8:24, 4:10] = 0.0     # 0.5 m: neighbouring pixels fall into neighbouring voxels
    for _ in range(5):                                                           # the anchor spreads a few voxels per cycle
        n_wall = _cycle(asm, wall, mask=robot).shape[0]
    mem = asm.static_memory
    born = (mem.tags & 1) != 0
    assert bool(born.any()) and not bool(born.all()) and bool(((mem.tags[born] & 2) != 0).all())     # some born, all of them anchored
    assert _cycle(asm, nothing, mask=all_robot).shape[0] == n_wall               # hidden by the arm: all held


def test_forgotten_rows_leave_the_memory():
    asm = _assembler()
    wall = np.full((H, W), 1.0, dtype=np.float32)
    n = _cycle(asm, wall).shape[0]
    # the consumer finds every second rest point of this cycle inside the robot: not taken from this observation either
    assert _cycle(asm, wall, forget=lambda p: torch.arange(0, p.shape[0], 2, device=DEV)).shape[0] == n
    assert asm.static_memory.size == n - (n + 1) // 2
    hidden = np.zeros((H, W), dtype=np.float32); robot = np.ones((H, W), dtype=bool)
    assert _cycle(asm, hidden, mask=robot).shape[0] == n - (n + 1) // 2      # what is held behind the robot: the rest only
    assert _cycle(asm, wall).shape[0] == n                                    # observed again: all there


def test_summary_counts():
    asm = _assembler(empty_frames_to_delete=1)
    wall = np.full((H, W), 1.0, dtype=np.float32)
    _cycle(asm, wall)
    _cycle(asm, np.full((H, W), 2.0, dtype=np.float32))
    text = asm.static_memory.summary()
    assert "deleted as empty" in text and "inserted" in text
