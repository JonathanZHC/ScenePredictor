from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam_rgbd_tracking.alignment import SharedWorldVoxelizer
from sam_rgbd_tracking.batched_postprocess import _MaskRecord
from sam_rgbd_tracking.config import Config
from sam_rgbd_tracking.data_types import CameraIntrinsics, RGBDFrame

CUDA = torch is not None and torch.cuda.is_available()


@unittest.skipUnless(CUDA, "CUDA required")
class GeometryStageCentroidTests(unittest.TestCase):
    def test_record_centroids_match_numpy_mean(self) -> None:
        from sam_rgbd_tracking.gpu_geometry import GPUSparseGeometryBackend

        cfg = Config({"shared_voxel_grid": {"voxel_size_m": 0.005, "origin_world": [0, 0, 0]}})
        backend = GPUSparseGeometryBackend(torch.device("cuda:0"), SharedWorldVoxelizer(cfg))
        H, W = 72, 96
        rng = np.random.default_rng(0)
        frames = []
        for v in range(2):
            yy, xx = np.mgrid[0:H, 0:W]
            depth = (1.0 + 0.002 * xx + 0.001 * yy + rng.normal(0, 0.001, (H, W))).astype(np.float32)
            pose = np.eye(4, dtype=np.float32); pose[0, 3] = 0.3 * v
            frames.append(RGBDFrame(f"cam{v}", 7, 1_000 + v, np.zeros((H, W, 3), np.uint8), depth,
                                    CameraIntrinsics(80.0, 80.0, W / 2, H / 2, W, H), pose))
        masks = []
        records = []
        boxes = [(0, 10, 40, 20, 60), (1, 5, 30, 50, 90), (1, 40, 70, 10, 40)]
        for r, (view, y0, y1, x0, x1) in enumerate(boxes):
            m = torch.zeros(H, W, dtype=torch.uint8, device="cuda"); m[y0:y1, x0:x1] = 1
            masks.append(m); records.append(_MaskRecord(view, r, r + 1))
        pending = backend.compute_from_masks(records, frames, masks, [1, 1, 1], max_points=4096,
                                             min_depth=0.1, max_depth=6.0)
        legacy = backend.materialize_compact(pending, records, frames, copy_points=True, copy_voxels=True)
        pending = backend.compute_from_masks(records, frames, masks, [1, 1, 1], max_points=4096,
                                             min_depth=0.1, max_depth=6.0)
        gpu_mode = backend.materialize_compact(pending, records, frames, copy_points=False, copy_voxels=False)
        torch.cuda.synchronize()
        for lg, gm in zip(legacy, gpu_mode):
            self.assertIsNotNone(lg.voxel_points)
            self.assertGreater(len(lg.voxel_points), 50)
            expected = lg.voxel_points.astype(np.float64).mean(0)
            np.testing.assert_allclose(lg.centroid_world, expected, atol=2e-5)
            np.testing.assert_allclose(gm.centroid_world, expected, atol=2e-5)
            # GPU mode: no host clouds, but device views and counts present
            self.assertIsNone(gm.points_world)
            self.assertIsNone(gm.voxel_points)
            self.assertIsNone(gm.voxel_keys)
            self.assertEqual(int(gm.voxel_points_gpu.shape[0]), len(lg.voxel_points))
            torch.testing.assert_close(gm.voxel_points_gpu.cpu(), torch.from_numpy(lg.voxel_points))


if __name__ == "__main__":
    unittest.main()
