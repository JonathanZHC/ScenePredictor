"""Fused visibility test of the static memory (one Triton kernel per camera view).

Per remembered point: camera transform p = R w + t, projection (u, v) = (fx x / z + cx, fy y / z + cy), the depth of
the pixel and the smallest valid depth of its (2h+1)^2 neighbourhood, margin(z) = max(min_margin, k (a + b z^2)), and
    seen   = inside & d_centre > 0 & |d_centre - z| <= margin
    empty  = inside & all valid neighbours beyond z + margin (at least one valid neighbour)
    occluded = inside & (robot pixel: occluder mask) or (d_centre valid & d_centre < z - margin)
    known  = seen | occluded | empty   (the rest is "unknown": no depth at the pixel, outside the image, behind the camera)
OR-ed into the int8 flag arrays over the views. Replaces ~15 torch kernels per view; the memory update was
launch-bound (its cost did not depend on the number of points).
"""
from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _visibility_kernel(
        points_ptr, depth_ptr, mask_ptr, seen_ptr, empty_ptr, known_ptr,
        n, height, width, fx, fy, cx, cy,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
        dmin, dmax, min_z, sig_a, sig_b, sig_k, min_margin,
        HAS_MASK: tl.constexpr, HALF: tl.constexpr, BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < n
        wx = tl.load(points_ptr + offs * 3 + 0, mask=inside, other=0.0)
        wy = tl.load(points_ptr + offs * 3 + 1, mask=inside, other=0.0)
        wz = tl.load(points_ptr + offs * 3 + 2, mask=inside, other=0.0)
        x = r00 * wx + r01 * wy + r02 * wz + t0
        y = r10 * wx + r11 * wy + r12 * wz + t1
        z = r20 * wx + r21 * wy + r22 * wz + t2
        front = z > min_z
        zs = tl.where(front, z, 1.0)
        uf = fx * x / zs + cx
        vf = fy * y / zs + cy
        u = tl.floor(uf + 0.5).to(tl.int32)
        v = tl.floor(vf + 0.5).to(tl.int32)
        in_img = inside & front & (u >= 0) & (u < width) & (v >= 0) & (v < height)
        uc = tl.minimum(tl.maximum(u, 0), width - 1)
        vc = tl.minimum(tl.maximum(v, 0), height - 1)
        d_c = tl.load(depth_ptr + vc * width + uc, mask=inside, other=0.0)
        d_min = tl.full([BLOCK], float("inf"), tl.float32)
        for dv in tl.static_range(-HALF, HALF + 1):
            for du in tl.static_range(-HALF, HALF + 1):
                un = tl.minimum(tl.maximum(u + du, 0), width - 1)
                vn = tl.minimum(tl.maximum(v + dv, 0), height - 1)
                dn = tl.load(depth_ptr + vn * width + un, mask=inside, other=0.0)
                ok = (dn >= dmin) & (dn <= dmax)
                d_min = tl.where(ok & (dn < d_min), dn, d_min)
        margin = tl.maximum(sig_k * (sig_a + sig_b * zs * zs), min_margin)
        c_valid = (d_c >= dmin) & (d_c <= dmax)
        seen = in_img & c_valid & (tl.abs(d_c - zs) <= margin)
        empty = in_img & (d_min != float("inf")) & (d_min > zs + margin)
        occluded = in_img & c_valid & (d_c < zs - margin)
        if HAS_MASK:
            robot = tl.load(mask_ptr + vc * width + uc, mask=inside, other=0)
            occluded = occluded | (in_img & (robot != 0))
        s_old = tl.load(seen_ptr + offs, mask=inside, other=0)
        e_old = tl.load(empty_ptr + offs, mask=inside, other=0)
        k_old = tl.load(known_ptr + offs, mask=inside, other=0)
        tl.store(seen_ptr + offs, tl.where(seen, 1, s_old).to(tl.int8), mask=inside)
        tl.store(empty_ptr + offs, tl.where(empty, 1, e_old).to(tl.int8), mask=inside)
        tl.store(known_ptr + offs, tl.where(seen | empty | occluded, 1, k_old).to(tl.int8), mask=inside)


def available(device: torch.device) -> bool:
    return triton is not None and device.type == "cuda"


def observe_view(points: torch.Tensor, depth: torch.Tensor, cam_from_world: "list[float]", intrinsics, depth_range,
                 min_z: float, margin: tuple[float, float, float, float], half: int,
                 seen: torch.Tensor, empty: torch.Tensor, known: torch.Tensor, mask: torch.Tensor | None = None) -> None:
    """OR the seen / empty / known flags of one view into the int8 [n] arrays (already allocated).

    cam_from_world: the 12 entries of the 3x4 camera<-world transform, row-major; intrinsics (fx, fy, cx, cy);
    margin (sigma_a, sigma_b, sigmas, min_margin)."""
    n = int(points.shape[0])
    if n == 0:
        return
    fx, fy, cx, cy = (float(v) for v in intrinsics)
    r = [float(v) for v in cam_from_world]
    grid = (triton.cdiv(n, _BLOCK),)
    has_mask = mask is not None
    _visibility_kernel[grid](
        points, depth, mask if has_mask else depth, seen, empty, known,
        n, int(depth.shape[0]), int(depth.shape[1]), fx, fy, cx, cy,
        r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9], r[10], r[3], r[7], r[11],
        float(depth_range[0]), float(depth_range[1]), float(min_z),
        float(margin[0]), float(margin[1]), float(margin[2]), float(margin[3]),
        HAS_MASK=has_mask, HALF=int(half), BLOCK=_BLOCK,
    )
