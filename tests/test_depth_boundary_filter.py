from __future__ import annotations

import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None

from sam_rgbd_tracking.batched_postprocess import BatchedPostprocessor, _MaskRecord
from sam_rgbd_tracking.config import Config

CUDA = torch is not None and torch.cuda.is_available()


# ----------------------------------------------------------------------------- NumPy reference
def erode4(m: np.ndarray, w: int) -> np.ndarray:
    m = m.copy()
    for _ in range(w):
        e = m.copy()
        e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        e[0, :] = e[-1, :] = False; e[:, 0] = e[:, -1] = False
        m = e
    return m


def grow(orig, core, depth, thr, steps):
    acc = core.copy()
    cand = orig & ~core
    for _ in range(steps):
        new = np.zeros_like(acc)
        new[:, 1:] |= acc[:, :-1] & (np.abs(depth[:, 1:] - depth[:, :-1]) < thr)
        new[:, :-1] |= acc[:, 1:] & (np.abs(depth[:, :-1] - depth[:, 1:]) < thr)
        new[1:, :] |= acc[:-1, :] & (np.abs(depth[1:, :] - depth[:-1, :]) < thr)
        new[:-1, :] |= acc[1:, :] & (np.abs(depth[:-1, :] - depth[1:, :]) < thr)
        acc |= new & cand
    return acc


def reference(masks, views, depth, thresholds, w):
    """Erosion + growing with given per-record thresholds (threshold < 0 -> passthrough)."""
    out = np.zeros_like(masks)
    for r in range(masks.shape[0]):
        d = depth[views[r]]
        orig = masks[r] & (d > 0)
        if thresholds[r] < 0:
            out[r] = orig
            continue
        out[r] = grow(orig, erode4(orig, w), d, thresholds[r], w)
    return out


def make_scene(H=120, W=160, seed=0):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    bg = 2.5 + 0.0004 * (xx - W / 2)
    depth = np.stack([bg.copy(), bg.copy() + 0.1])
    masks, views = [], []
    for v in range(2):
        sphere = (xx - 50) ** 2 + (yy - 60) ** 2 < 35 ** 2
        z = 1.2 - 0.1 * np.sqrt(np.clip(1 - ((xx - 50) ** 2 + (yy - 60) ** 2) / 35 ** 2, 0, 1))
        depth[v] = np.where(sphere, z, depth[v])
        slab = (xx > 95) & (xx < 140) & (yy > 20) & (yy < 100)
        depth[v] = np.where(slab, 2.2 + 0.001 * (xx - 95), depth[v])
        thin = (xx >= 75) & (xx < 80) & (yy > 10) & (yy < 110)   # 5 px: no core -> fallback
        depth[v] = np.where(thin, 1.5, depth[v])
        for m in (sphere, slab, thin):
            ring = m & ~erode4(m, 2)
            fly = ring & (rng.random((H, W)) < 0.5)
            depth[v] = np.where(fly, 0.5 * depth[v] + 0.5 * bg, depth[v])
            masks.append(m); views.append(v)
    depth += rng.normal(0, 0.002, depth.shape).astype(np.float32)
    depth[rng.random(depth.shape) < 0.003] = 0.0
    return np.stack(masks), views, depth.astype(np.float32)


@unittest.skipUnless(CUDA, "CUDA required")
class DepthBoundaryFilterKernelTests(unittest.TestCase):
    def test_matches_numpy_reference_two_views(self) -> None:
        from sam_rgbd_tracking.depth_boundary_filter import DepthBoundaryFilter

        masks, views, depth = make_scene()
        filt = DepthBoundaryFilter(torch.device("cuda"), erosion_width_px=5, mad_multiplier=5.0)
        out = filt(
            torch.from_numpy(masks).cuda().to(torch.uint8).contiguous(),
            views,
            torch.from_numpy(depth).cuda().contiguous(),
        )
        torch.cuda.synchronize()
        stats = filt.last_stats()[: masks.shape[0]].cpu().numpy()
        out_np = out.cpu().numpy().astype(bool)

        # sphere + slab get real thresholds; the 5-px thin bar has no core -> fallback
        self.assertTrue(all(stats[r, 0] > 0 for r in (0, 1, 3, 4)))
        self.assertTrue(all(stats[r, 0] < 0 for r in (2, 5)))
        ref = reference(masks, views, depth, stats[:, 0], 5)
        self.assertEqual(int((out_np != ref).sum()), 0)

        # flyers removed on the sphere, valid surface mostly kept, thin bar untouched
        core = erode4(masks[0] & (depth[0] > 0), 5)
        self.assertGreater(out_np[0].sum(), 0.9 * masks[0].sum())
        self.assertTrue((out_np[0] & core).sum() == core.sum())
        self.assertTrue(np.array_equal(out_np[2], masks[2] & (depth[0] > 0)))

    def test_second_call_reuses_buffers_and_is_deterministic(self) -> None:
        from sam_rgbd_tracking.depth_boundary_filter import DepthBoundaryFilter

        masks, views, depth = make_scene(seed=3)
        filt = DepthBoundaryFilter(torch.device("cuda"))
        m = torch.from_numpy(masks).cuda().to(torch.uint8).contiguous()
        d = torch.from_numpy(depth).cuda().contiguous()
        a = filt(m, views, d).clone()
        b = filt(m, views, d).clone()
        self.assertTrue(torch.equal(a, b))


@unittest.skipUnless(CUDA, "CUDA required")
class DepthBoundaryFilterIntegrationTests(unittest.TestCase):
    def test_batch_masks_gpu_applies_filter_to_tracked_only(self) -> None:
        from sam_rgbd_tracking.depth_boundary_filter import DepthBoundaryFilter

        config = Config(
            {
                "runtime": {"device": "cuda"},
                "detector": {"excluded_labels": ["robot"]},
                "postprocess": {
                    "mask_threshold": 0.0,
                    "tracking_erosion_pixels": 0,
                    "exclusion_dilation_pixels": 1,
                    "gpu_batch": True,
                    "gpu_geometry": False,
                    "cpu_workers": 1,
                },
            }
        )
        post = BatchedPostprocessor(config, 2)
        try:
            masks, views, depth = make_scene()
            depth_gpu = torch.from_numpy(depth).cuda().contiguous()
            # Inject the filter + depth source (gpu_geometry is off in this unit test).
            post._depth_boundary_filter = DepthBoundaryFilter(torch.device("cuda"))
            post._depth_gpu_for_mask_filter = lambda frames: depth_gpu

            H, W = depth.shape[1:]
            records = []
            logits = [np.full((4, H, W), -1.0, np.float32) for _ in range(2)]
            channel = [0, 0]
            for r in range(masks.shape[0]):
                v = views[r]
                logits[v][channel[v]][masks[r]] = 1.0
                records.append(_MaskRecord(v, channel[v], r + 1, excluded=False))
                channel[v] += 1
            # one excluded record per view: a 1-px dot that must be dilated, not filtered
            for v in range(2):
                logits[v][channel[v]][5, 5] = 1.0
                records.append(_MaskRecord(v, channel[v], 100 + v, excluded=True))
                channel[v] += 1

            pending, exclusion = post._batch_masks_gpu(
                records,
                {(H, W, H, W): list(range(len(records)))},
                [torch.from_numpy(l).cuda() for l in logits],
                need_raw_masks=False,
                need_final_masks=True,
                frames=[object(), object()],
            )
            torch.cuda.synchronize()
            self.assertEqual(len(pending), masks.shape[0])
            stats = post._depth_boundary_filter.last_stats()[: masks.shape[0]].cpu().numpy()
            ref = reference(masks, views, depth, stats[:, 0], 5)
            for (record, cpu_mask), r in zip(pending, range(masks.shape[0])):
                got = record.final_mask_gpu.cpu().numpy().astype(bool)
                self.assertEqual(int((got != ref[r]).sum()), 0, f"record {r}")
                self.assertTrue(np.array_equal(cpu_mask.astype(bool), ref[r]))
            for v in range(2):
                self.assertEqual(int(exclusion[v].sum().item()), 9)
        finally:
            post.close()


if __name__ == "__main__":
    unittest.main()
