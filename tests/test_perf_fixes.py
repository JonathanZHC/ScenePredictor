from __future__ import annotations

import unittest


try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam_rgbd_tracking.config import Config

CUDA = torch is not None and torch.cuda.is_available()


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
