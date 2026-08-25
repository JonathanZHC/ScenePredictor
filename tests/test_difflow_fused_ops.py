from __future__ import annotations

import unittest

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

try:
    from difflow3d.ops import fused_cross_block, fused_knn
    from difflow3d.model.pointconv import group_channel_first
    HAVE_DIFFLOW = True
except Exception:  # pragma: no cover - DifFlow3D not on sys.path in this environment
    HAVE_DIFFLOW = False

CUDA = torch is not None and torch.cuda.is_available()


@unittest.skipUnless(CUDA and HAVE_DIFFLOW, "CUDA + difflow3d required")
class FusedKnnTests(unittest.TestCase):
    def test_exact_against_fp64_brute_force(self) -> None:
        g = torch.Generator(device="cuda").manual_seed(0)
        for n, m, k in [(1024, 1024, 16), (1024, 1024, 3), (512, 512, 32), (300, 1000, 3)]:
            q = torch.rand(1, n, 3, device="cuda", generator=g) * 2 - 1
            r = torch.rand(1, m, 3, device="cuda", generator=g) * 2 - 1
            idx = fused_knn.knn_indices(q, r, k)
            self.assertEqual(idx.shape, (1, n, k))
            self.assertEqual(idx.dtype, torch.int64)
            ref_d, _ = torch.topk(torch.cdist(q.double(), r.double()) ** 2, k, dim=-1, largest=False)
            got_d = ((q.double().unsqueeze(2) - r.double()[0][idx]) ** 2).sum(-1)
            torch.testing.assert_close(torch.sort(got_d, -1).values, torch.sort(ref_d, -1).values, atol=1e-9, rtol=0)
            # ascending order, self included for self-KNN
            self.assertTrue((got_d[..., 1:] >= got_d[..., :-1] - 1e-12).all())
        p = torch.rand(1, 256, 3, device="cuda", generator=g)
        self.assertTrue((fused_knn.knn_indices(p, p, 9)[..., 0] == torch.arange(256, device="cuda")).all())

    def test_graph_capture(self) -> None:
        q = torch.rand(1, 1024, 3, device="cuda"); r = torch.rand(1, 1024, 3, device="cuda")
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = fused_knn.knn_indices(q, r, 16)
        q.copy_(torch.rand(1, 1024, 3, device="cuda"))
        graph.replay(); torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, fused_knn.knn_indices(q, r, 16)))


@unittest.skipUnless(CUDA and HAVE_DIFFLOW, "CUDA + difflow3d required")
class FusedCrossBlockTests(unittest.TestCase):
    def test_matches_eager(self) -> None:
        torch.manual_seed(0)
        tf32 = torch.backends.cudnn.allow_tf32
        torch.backends.cudnn.allow_tf32 = False
        try:
            for c, n, m, k in [(128, 512, 512, 32), (32, 1024, 1024, 32), (256, 256, 256, 32), (64, 7, 13, 5)]:
                pos = torch.nn.Conv2d(3, c, 1).cuda().eval()
                relu = torch.nn.LeakyReLU(0.1)
                p1 = torch.randn(1, c, n, device="cuda"); p2 = torch.randn(1, c, m, device="cuda")
                idx = torch.randint(0, m, (1, n, k), device="cuda", dtype=torch.int32)
                d = torch.randn(1, 3, n, k, device="cuda") * 0.1
                with torch.inference_mode():
                    self.assertTrue(fused_cross_block.supported(pos, torch.nn.Identity(), relu, p1, p2, idx, d))
                    ref = relu(group_channel_first(p2, idx) + p1.unsqueeze(3) + pos(d))
                    got = fused_cross_block.cross_prologue(pos, relu, p1, p2, idx, d)
                torch.testing.assert_close(got, ref, atol=2e-6, rtol=1e-6)
        finally:
            torch.backends.cudnn.allow_tf32 = tf32

    def test_unsupported_configs_fall_back(self) -> None:
        pos = torch.nn.Conv2d(3, 8, 1).cuda()
        p1 = torch.randn(1, 8, 4, device="cuda"); p2 = torch.randn(1, 8, 6, device="cuda")
        idx = torch.randint(0, 6, (1, 4, 3), device="cuda", dtype=torch.int32)
        d = torch.randn(1, 3, 4, 3, device="cuda")
        self.assertFalse(fused_cross_block.supported(pos, torch.nn.BatchNorm2d(8).cuda(), torch.nn.ReLU(), p1, p2, idx, d))
        self.assertFalse(fused_cross_block.supported(pos, torch.nn.Identity(), torch.nn.GELU(), p1, p2, idx, d))


if __name__ == "__main__":
    unittest.main()
