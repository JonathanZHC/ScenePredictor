"""Fused rest-scene backprojection (one Triton kernel per camera view).

Per pixel: depth validity (finite, inside [min, max], not excluded), backprojection with the rays
x = (col - cx) / fx, y = (row - cy) / fy, world transform p = R [x z, y z, z] + t in plain fp32 (no GEMM, so no TF32:
the former `points @ R.T` was off by up to 1.4 mm at 3 m with TF32 enabled), and the 1-cell voxel key of the shared
lattice ((sx << 42) | (sy << 21) | sz, sx = floor((p - origin) / voxel) + 2^20; INT64_MAX = invalid). It replaces
~60 small torch kernels per view (the rest-scene assembly was launch-bound: 1.9 ms CPU for 0.6 ms of GPU work).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

INVALID_KEY = torch.iinfo(torch.int64).max
_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _rest_scene_kernel(
        depth_ptr, excl_ptr, points_ptr, keys_ptr,
        n, width, fx, fy, cx, cy,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
        ox, oy, oz, inv_voxel, dmin, dmax,
        HAS_EXCL: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < n
        d = tl.load(depth_ptr + offs, mask=inside, other=0.0)
        valid = inside & (d == d) & (tl.abs(d) != float("inf")) & (d >= dmin) & (d <= dmax)
        if HAS_EXCL:
            ex = tl.load(excl_ptr + offs, mask=inside, other=1)
            valid = valid & (ex == 0)
        z = tl.where(valid, d, 0.0)
        row = (offs // width).to(tl.float32)
        col = (offs % width).to(tl.float32)
        x = (col - cx) / fx * z
        y = (row - cy) / fy * z
        px = r00 * x + r01 * y + r02 * z + t0
        py = r10 * x + r11 * y + r12 * z + t1
        pz = r20 * x + r21 * y + r22 * z + t2
        tl.store(points_ptr + offs * 3 + 0, px, mask=inside)
        tl.store(points_ptr + offs * 3 + 1, py, mask=inside)
        tl.store(points_ptr + offs * 3 + 2, pz, mask=inside)
        sx = tl.floor((px - ox) * inv_voxel).to(tl.int64) + 1048576
        sy = tl.floor((py - oy) * inv_voxel).to(tl.int64) + 1048576
        sz = tl.floor((pz - oz) * inv_voxel).to(tl.int64) + 1048576
        key_max = 2097151
        in_range = (sx >= 0) & (sx <= key_max) & (sy >= 0) & (sy <= key_max) & (sz >= 0) & (sz <= key_max)
        key = (sx << 42) | (sy << 21) | sz
        key = tl.where(valid & in_range, key, 9223372036854775807)
        tl.store(keys_ptr + offs, key, mask=inside)


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


def backproject_view(
    depth: torch.Tensor,
    excluded: torch.Tensor | None,
    intrinsics: tuple[float, float, float, float],
    transform: "list[list[float]]",
    origin: "tuple[float, float, float]",
    inv_voxel: float,
    depth_range: tuple[float, float],
    points_out: torch.Tensor,
    keys_out: torch.Tensor,
) -> None:
    """depth [H*W] float32 CUDA (metres), excluded [H*W] uint8/bool or None; writes points_out [H*W,3] and keys_out [H*W]."""
    n = int(depth.numel())
    fx, fy, cx, cy, width = intrinsics
    (r00, r01, r02, t0), (r10, r11, r12, t1), (r20, r21, r22, t2) = transform[:3]
    excl = excluded.view(torch.uint8) if excluded is not None and excluded.dtype == torch.bool else excluded
    grid = (triton.cdiv(n, _BLOCK),)
    _rest_scene_kernel[grid](
        depth, excl if excl is not None else depth, points_out, keys_out,
        n, int(width), float(fx), float(fy), float(cx), float(cy),
        float(r00), float(r01), float(r02), float(r10), float(r11), float(r12), float(r20), float(r21), float(r22),
        float(t0), float(t1), float(t2),
        float(origin[0]), float(origin[1]), float(origin[2]), float(inv_voxel),
        float(depth_range[0]), float(depth_range[1]),
        HAS_EXCL=excl is not None, BLOCK=_BLOCK,
    )
