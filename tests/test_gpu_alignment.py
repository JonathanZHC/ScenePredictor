from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam_rgbd_tracking.alignment import (
    CrossFrameAligner,
    CrossViewAligner,
    SharedWorldVoxelizer,
    gpu_alignment_enabled,
)
from sam_rgbd_tracking.config import Config
from sam_rgbd_tracking.data_types import FrameResult, ProcessedInstance, VisibilityState

CUDA = torch is not None and torch.cuda.is_available()


def _config(gpu_alignment: bool) -> Config:
    return Config(
        {
            "runtime": {"device": "cuda:0", "enable_visualization": False, "camera_names": ["cam0", "cam1"]},
            "detector": {"prompts": [["ball", 1], ["mug", 1], ["bottle", 1]]},
            "postprocess": {"gpu_geometry": True, "gpu_alignment": gpu_alignment},
            "shared_voxel_grid": {"voxel_size_m": 0.005, "origin_world": [0, 0, 0], "match_radius_voxels": 1,
                                  "min_alignment_score": 0.15, "min_bidirectional_coverage": 0.2,
                                  "max_local_dense_voxels": 1_000_000},
            "cross_frame_alignment": {"centroid_gate_m": 0.2, "chamfer_max_workspace_mb": 24,
                                      "chamfer_preallocate_points": 8192},
        }
    )


def _cloud(rng, center, n, spread=0.06):
    return (center + rng.normal(0, spread, (n, 3))).astype(np.float32)


class _Frame:
    def __init__(self, name):
        self.camera_name = name


def _instance(vox: SharedWorldVoxelizer, label: str, track_id: int, points: np.ndarray, *, gpu: bool, cpu_points: bool):
    coords, keys, upoints, _, bmin, bmax = vox.prepare_points(points, None)
    return ProcessedInstance(
        track_id=track_id, label=label, semantic_confidence=1.0, tracking_confidence=1.0,
        motion_prediction_confidence=1.0, raw_mask=np.empty((0, 0), bool), mask=np.empty((0, 0), bool),
        points_camera=np.empty((0, 3), np.float32), points_world=upoints if cpu_points else None,
        colors_rgb=np.empty((0, 3), np.uint8), centroid_camera=None, centroid_world=None,
        bbox_min=None, bbox_max=None, bbox_2d_xyxy=(0, 0, 1, 1), status=VisibilityState.VISIBLE,
        voxel_coords=coords, voxel_keys=keys, voxel_points=upoints if cpu_points else None,
        voxel_bbox_min=bmin, voxel_bbox_max=bmax,
        voxel_points_gpu=torch.from_numpy(upoints).cuda() if gpu else None,
    )


def _frames(vox, rng, t, *, gpu, cpu_points):
    """Two cameras: 'ball' seen by both (overlapping surface), 'mug' only by cam1."""
    ball = np.array([0.5 + 0.02 * t, 0.1, 0.3])
    shared = _cloud(rng, ball, 3000)
    view0 = [_instance(vox, "ball", 1, shared[:2000], gpu=gpu, cpu_points=cpu_points)]
    view1 = [
        _instance(vox, "ball", 1, shared[800:], gpu=gpu, cpu_points=cpu_points),
        _instance(vox, "mug", 2, _cloud(rng, np.array([0.9, -0.2, 0.25 + 0.01 * t]), 1500), gpu=gpu, cpu_points=cpu_points),
    ]
    return [
        FrameResult(frame=_Frame("cam0"), instances=view0, exclusion_mask_gpu=None, owner_track_map=None, keyframe=False, timings_ms={}),
        FrameResult(frame=_Frame("cam1"), instances=view1, exclusion_mask_gpu=None, owner_track_map=None, keyframe=False, timings_ms={}),
    ]


@unittest.skipUnless(CUDA, "CUDA required")
class GpuAlignmentEquivalenceTests(unittest.TestCase):
    def test_flag(self) -> None:
        self.assertTrue(gpu_alignment_enabled(_config(True)))
        self.assertFalse(gpu_alignment_enabled(_config(False)))

    def test_two_views_two_frames_match_legacy(self) -> None:
        legacy_cfg, gpu_cfg = _config(False), _config(True)
        legacy_cv, gpu_cv = CrossViewAligner(legacy_cfg), CrossViewAligner(gpu_cfg)
        legacy_cf, gpu_cf = CrossFrameAligner(legacy_cfg, num_views=2), CrossFrameAligner(gpu_cfg, num_views=2)
        self.assertFalse(legacy_cv.gpu_fusion)
        self.assertTrue(gpu_cv.gpu_fusion)

        for t in range(3):
            rng_a, rng_b = np.random.default_rng(t), np.random.default_rng(t)
            legacy_frames = _frames(legacy_cv.voxelizer, rng_a, t, gpu=False, cpu_points=True)
            gpu_frames = _frames(gpu_cv.voxelizer, rng_b, t, gpu=True, cpu_points=False)

            legacy_groups, lc = legacy_cv.align(legacy_frames)
            gpu_groups, gc = gpu_cv.align(gpu_frames)
            for key in ("num_cross_view_candidate_pairs", "num_cross_view_matches", "num_fused_points_before_downsample"):
                self.assertEqual(lc[key], gc[key], key)
            # GPU mode (no cross-view dedup) keeps every member point
            self.assertEqual(gc["num_fused_points_after_downsample"], gc["num_fused_points_before_downsample"])
            self.assertEqual(len(legacy_groups), len(gpu_groups))
            self.assertEqual(len(legacy_groups), 2)  # ball fused across views + mug
            for lg, gg in zip(legacy_groups, gpu_groups):
                self.assertEqual([c for c, _ in lg.members], [c for c, _ in gg.members])
                self.assertIsNone(gg.points_world)
                members = [inst for _, inst in gg.members]
                if len(members) == 1:
                    self.assertEqual(lg.point_count, gg.point_count)
                    self.assertIsNotNone(gg.fused_points_gpu)
                    fused_gpu = gg.fused_points_gpu.cpu().numpy()
                    a = np.lexsort(lg.points_world.T); b = np.lexsort(fused_gpu.T)
                    np.testing.assert_array_equal(lg.points_world[a], fused_gpu[b])
                    # no record means in this synthetic setup -> GPU median fallback == legacy median
                    np.testing.assert_allclose(gg.centroid_world, lg.centroid_world, atol=1e-5)
                else:
                    # Default GPU mode concatenates member clouds (no cross-view
                    # dedup): count is the sum, legacy count is the voxel union.
                    self.assertEqual(gg.point_count, sum(int(m.voxel_points_gpu.shape[0]) for m in members))
                    self.assertGreaterEqual(gg.point_count, lg.point_count)
                    # With record means (production) the members stay a list; without
                    # them (this synthetic setup) the fallback concatenates on device.
                    self.assertTrue(gg.fused_member_clouds_gpu is not None or gg.fused_points_gpu is not None)
                    # centroid of the concatenation vs legacy median of the union: same object
                    np.testing.assert_allclose(gg.centroid_world, lg.centroid_world, atol=0.02)

            legacy_cf.align(legacy_groups)
            gpu_cf.align(gpu_groups)
            torch.cuda.synchronize()
            self.assertEqual([g.global_track_id for g in legacy_groups], [g.global_track_id for g in gpu_groups])
            gw = gpu_cf.workspace
            for g, count in zip(gpu_groups, gw.current_counts_np):
                self.assertEqual(int(count), g.point_count)
                self.assertIsNotNone(g.points_world_gpu)
                self.assertEqual(int(g.points_world_gpu.shape[0]), g.point_count)
                # bank row == concatenation of the member clouds (or the single cloud)
                parts = [inst.voxel_points_gpu for _, inst in g.members]
                torch.testing.assert_close(g.points_world_gpu, torch.cat(parts, 0))

    def test_single_view_gpu_only_observation(self) -> None:
        cfg = _config(True)
        cv = CrossViewAligner(cfg)
        rng = np.random.default_rng(0)
        frames = _frames(cv.voxelizer, rng, 0, gpu=True, cpu_points=False)[1:]  # cam1 only
        # single camera: strip CPU voxel data entirely, as materialize_compact would
        for inst in frames[0].instances:
            inst.voxel_coords = inst.voxel_keys = None
            inst.voxel_bbox_min = inst.voxel_bbox_max = None
        groups, counters = cv.align(frames)
        self.assertEqual(len(groups), 2)
        self.assertEqual(counters["num_cross_view_matches"], 0)
        for g in groups:
            self.assertGreater(g.point_count, 0)
            self.assertIsNotNone(g.centroid_world)
            expected = np.median(g.fused_points_gpu.cpu().numpy(), axis=0)
            np.testing.assert_allclose(g.centroid_world, expected, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
