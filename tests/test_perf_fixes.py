from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam_rgbd_tracking.alignment import SharedWorldVoxelizer, _VoxelObservation
from sam_rgbd_tracking.config import Config

CUDA = torch is not None and torch.cuda.is_available()


def _observation(voxelizer: SharedWorldVoxelizer, coords: np.ndarray, view: int) -> _VoxelObservation:
    coords = np.unique(np.asarray(coords, dtype=np.int64), axis=0)
    keys = voxelizer._encode_keys(coords)
    order = np.argsort(keys)
    return _VoxelObservation(
        view_index=view,
        camera_name=f"cam{view}",
        instance=None,  # type: ignore[arg-type]
        coords=coords[order],
        keys=keys[order],
        points=coords[order].astype(np.float32) * voxelizer.voxel_size_m,
        colors=np.zeros((len(coords), 3), np.uint8),
        bbox_min=coords.min(axis=0),
        bbox_max=coords.max(axis=0),
    )


class LazyCoverageTests(unittest.TestCase):
    def test_coverage_grid_built_lazily_and_matches_sparse(self) -> None:
        cfg = Config(
            {
                "shared_voxel_grid": {
                    "voxel_size_m": 0.01,
                    "match_radius_voxels": 1,
                    "max_local_dense_voxels": 1_000_000,
                }
            }
        )
        vox = SharedWorldVoxelizer(cfg)
        rng = np.random.default_rng(0)
        a = _observation(vox, rng.integers(0, 20, (300, 3)), 0)
        b = _observation(vox, rng.integers(0, 20, (300, 3)), 1)
        # Not built at construction (this was the 3-5 ms per-frame cost).
        self.assertFalse(a.coverage_built)
        self.assertIsNone(a.coverage_grid)

        dense = vox.directional_coverage(a, b)
        self.assertTrue(b.coverage_built)
        self.assertIsNotNone(b.coverage_grid)
        self.assertFalse(a.coverage_built)  # only the target needs it

        # Same answer as the sparse fallback path.
        vox_sparse = SharedWorldVoxelizer(
            Config({"shared_voxel_grid": {"voxel_size_m": 0.01, "match_radius_voxels": 1, "max_local_dense_voxels": 0}})
        )
        b2 = _observation(vox_sparse, b.coords, 1)
        sparse = vox_sparse.directional_coverage(a, b2)
        self.assertAlmostEqual(dense, sparse, places=6)


@unittest.skipUnless(CUDA, "CUDA required")
class DirectBufferCapacityTests(unittest.TestCase):
    def test_ensure_direct_buffers_is_grow_only(self) -> None:
        from sam_rgbd_tracking.alignment import SharedWorldVoxelizer
        from sam_rgbd_tracking.gpu_geometry import GPUSparseGeometryBackend

        cfg = Config({"shared_voxel_grid": {"voxel_size_m": 0.005}})
        backend = GPUSparseGeometryBackend(torch.device("cuda:0"), SharedWorldVoxelizer(cfg))
        backend._ensure_direct_buffers(4, 2, 360, 640, 4096)
        masks_id = backend._direct_masks.data_ptr()
        # Fewer records / same views: no reallocation.
        backend._ensure_direct_buffers(2, 2, 360, 640, 4096)
        self.assertEqual(backend._direct_masks.data_ptr(), masks_id)
        self.assertEqual(backend._direct_shape[0], 4)
        # More records: grows, keeps view capacity.
        backend._ensure_direct_buffers(6, 1, 360, 640, 2048)
        self.assertEqual(backend._direct_shape[:2], (6, 2))
        self.assertEqual(backend._direct_max_points, 4096)
        self.assertEqual(backend._direct_points_camera.shape[0], 6 * 4096)


if __name__ == "__main__":
    unittest.main()
