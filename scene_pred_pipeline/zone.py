"""Zone of the pipeline: a box outside which nothing is computed.

`zone: {min, max, pose}` in the pipeline config is a box in its own frame (pose = [t_xyz, axis-angle] of that frame in
the world; identity when omitted - an application that knows the robot puts the robot base there). At the start of
every cycle the depth of each camera is set to 0 (= no measurement) wherever its world point lies outside the box, so
every later stage - tracker lifting, flow, rest scene, static memory - works on the points inside the zone only and
nothing downstream needs a box test of its own. Without a zone (the default) all points are used.

One consumer needs to know what lies BEYOND the zone as well: the static memory decides that a remembered point is gone
when a camera ray passes through it, and that ray usually ends on a surface outside the zone. The unrestricted depth
image therefore stays on the frame as `depth_unzoned_gpu` and the memory's visibility test reads that one.

Cost: one fused kernel per camera on the depth image that is already on the device (the self-filter hands it over
there); a host depth image is uploaded once and the later stages then use the device copy instead of uploading it
themselves.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .data_types import LazyHostDepth, MultiCameraFrame

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover
    triton = None
    tl = None

_BLOCK = 1024

if triton is not None:

    @triton.jit
    def _zone_kernel(
        depth_ptr, out_ptr, n, width, fx, fy, cx, cy,
        r00, r01, r02, r10, r11, r12, r20, r21, r22, t0, t1, t2,
        lx, ly, lz, hx, hy, hz,
        BLOCK: tl.constexpr,
    ):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        inside = offs < n
        z = tl.load(depth_ptr + offs, mask=inside, other=0.0)
        row = (offs // width).to(tl.float32)
        col = (offs % width).to(tl.float32)
        x = (col - cx) / fx * z
        y = (row - cy) / fy * z
        px = r00 * x + r01 * y + r02 * z + t0
        py = r10 * x + r11 * y + r12 * z + t1
        pz = r20 * x + r21 * y + r22 * z + t2
        keep = (px >= lx) & (px <= hx) & (py >= ly) & (py <= hy) & (pz >= lz) & (pz <= hz)     # NaN / inf depth: not kept
        tl.store(out_ptr + offs, tl.where(keep, z, 0.0), mask=inside)


def axis_angle_matrix(rotvec) -> np.ndarray:
    r = np.asarray(rotvec, dtype=np.float64).reshape(3)
    angle = float(np.linalg.norm(r))
    if angle < 1e-12:
        return np.eye(3)
    k = r / angle
    K = np.array([[0.0, -k[2], k[1]], [k[2], 0.0, -k[0]], [-k[1], k[0], 0.0]])
    return np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)


class ZoneFilter:
    def __init__(self, config: Any, device: torch.device) -> None:
        self.device = torch.device(device)
        self.lo = tuple(float(v) for v in config.min)
        self.hi = tuple(float(v) for v in config.max)
        pose = np.asarray(config.pose, dtype=np.float64).reshape(6)
        world_from_zone = np.eye(4)
        world_from_zone[:3, :3] = axis_angle_matrix(pose[3:])
        world_from_zone[:3, 3] = pose[:3]
        self.zone_from_world = np.linalg.inv(world_from_zone)
        self.use_kernel = triton is not None and self.device.type == "cuda"
        self._cams: dict[str, tuple[bytes, list[float]]] = {}
        self._rays: dict[tuple, tuple[torch.Tensor, torch.Tensor]] = {}

    def describe(self) -> str:
        return (f"zone x {self.lo[0]:+.2f}..{self.hi[0]:+.2f}, y {self.lo[1]:+.2f}..{self.hi[1]:+.2f}, "
                f"z {self.lo[2]:+.2f}..{self.hi[2]:+.2f} m in the zone frame")

    def _transform(self, name: str, world_from_camera) -> list[float]:
        T = np.eye(4)
        wc = np.asarray(world_from_camera, dtype=np.float64)
        T[: wc.shape[0], :4] = wc[:, :4]
        key = T.tobytes()
        cached = self._cams.get(name)
        if cached is None or cached[0] != key:
            cached = (key, [float(v) for v in (self.zone_from_world @ T)[:3, :4].reshape(-1)])
            self._cams[name] = cached
        return cached[1]

    def restrict(self, depth: torch.Tensor, K, zone_from_camera: list[float]) -> torch.Tensor:
        """Copy of the depth image [H, W] with 0 wherever the pixel's point lies outside the zone."""
        K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
        r = zone_from_camera
        height, width = int(depth.shape[0]), int(depth.shape[1])
        if self.use_kernel:
            depth = depth.contiguous()
            out = torch.empty_like(depth)
            n = height * width
            _zone_kernel[(triton.cdiv(n, _BLOCK),)](
                depth, out, n, width, fx, fy, cx, cy,
                r[0], r[1], r[2], r[4], r[5], r[6], r[8], r[9], r[10], r[3], r[7], r[11],
                self.lo[0], self.lo[1], self.lo[2], self.hi[0], self.hi[1], self.hi[2], BLOCK=_BLOCK,
            )
            return out
        key = (height, width, fx, fy, cx, cy)
        rays = self._rays.get(key)
        if rays is None:
            rows, cols = torch.meshgrid(torch.arange(height, device=depth.device, dtype=torch.float32),
                                        torch.arange(width, device=depth.device, dtype=torch.float32), indexing="ij")
            rays = ((cols - cx) / fx, (rows - cy) / fy)
            self._rays = {key: rays}
        x, y = rays[0] * depth, rays[1] * depth
        keep = torch.ones_like(depth, dtype=torch.bool)
        for axis in range(3):
            p = r[4 * axis] * x + r[4 * axis + 1] * y + r[4 * axis + 2] * depth + r[4 * axis + 3]
            keep &= (p >= self.lo[axis]) & (p <= self.hi[axis])
        return torch.where(keep, depth, torch.zeros_like(depth))

    def apply(self, frame: MultiCameraFrame) -> None:
        """Restrict the depth of every camera of the frame in place (depth, depth_gpu); the image as it came stays in
        depth_unzoned_gpu."""
        for name, cam in frame.cameras.items():
            depth = getattr(cam, "depth_gpu", None)
            if depth is None:
                host = np.ascontiguousarray(np.asarray(cam.depth), dtype=np.float32)
                if host.ndim != 2 or host.size == 0:
                    continue
                depth = torch.as_tensor(host).to(self.device)
            elif depth.ndim != 2 or depth.numel() == 0:
                continue
            world_from_camera = np.asarray(cam.T_world_camera)
            if world_from_camera.shape not in {(4, 4), (3, 4)}:
                continue
            zoned = self.restrict(depth, cam.K, self._transform(name, world_from_camera))
            cam.depth_unzoned_gpu = depth
            cam.depth_gpu = zoned
            cam.depth = LazyHostDepth(zoned)
