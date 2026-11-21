"""Visibility-aware memory of the static (rest) scene.

The rest scene is rebuilt from the depth images every cycle, so a static surface that the robot or another
object hides from both cameras vanishes from the cloud handed to the safety filter - exactly when the robot
is close to it. This module keeps every rest point that was seen once, in the world frame, until a camera
ray proves that its place is empty:

    for each remembered point and each camera, project it into the (filtered) depth image and compare the
    projected depth z with the observed depth d of the pixel (3x3 neighbourhood):
        seen      |d - z| <= margin(z)                 -> refreshed by the current rest point of its voxel
        occluded  d < z - margin (something in front), or the pixel is in the camera's occluder mask (the robot,
                  removed by the self-filter)            -> kept as it is, for as long as it lasts
        empty     all valid neighbours have d > z + margin (the ray passed through the point) -> delete
                  after `empty_frames_to_delete` consecutive cycles
        unknown   no evidence from any camera (holes, out of range, outside the image) -> kept for
                  `unknown_frames_to_delete` consecutive cycles, then dropped

Why unknown expires: a transient object in front of a background without depth return (a window, the far field,
a stereo hole) would otherwise be remembered for ever. Robot pixels are the one case where "no depth" must keep the
point indefinitely, and the self-filter knows which pixels those are (CameraFrameCpu.occluder_mask). One more rule keeps the memory close to the current scene: a point that is
seen (the ray hits a surface at its depth) but whose voxel has not produced a rest point for
`seen_unrefreshed_frames_to_delete` cycles is dropped - it was a depth-noise voxel next to the surface, and the
surface itself is still represented by the refreshed voxels. Points are matched to the current rest scene on the rest-scene voxel
lattice (same origin and voxel size), so a voxel holds at most one point: the current observation when the voxel
is observed this cycle, else the remembered one.

Static-only operation: no motion model, remembered points have zero velocity. Moving objects must be handled by the
tracked pipeline; with tracked classes the rest scene already excludes their masks.

margin(z) = max(min_margin_m, margin_sigmas * (depth_sigma_a_m + depth_sigma_b_per_m * z^2)), a depth-noise model
of the stereo camera; min_margin_m defaults to the voxel size (a voxel centre is up to half a voxel off the surface,
and surfaces seen at grazing angles spread over neighbouring pixels).
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import rest_scene_kernel, static_memory_kernel

KEY_BIAS = 1 << 20
KEY_MASK = (1 << 21) - 1


class StaticMemory:
    def __init__(
        self,
        config: Any,
        *,
        device: torch.device,
        origin: torch.Tensor,
        voxel_size_m: float,
        depth_range: tuple[float, float],
        workspace: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> None:
        self.cfg = config
        self.device = device
        self._origin = origin.to(device=device, dtype=torch.float32)
        self._inv_voxel = 1.0 / float(voxel_size_m)
        self._min_depth, self._max_depth = float(depth_range[0]), float(depth_range[1])
        self._workspace = workspace
        self._min_margin = float(config.min_margin_m) if config.min_margin_m > 0.0 else float(voxel_size_m)
        self._half = int(config.neighbourhood) // 2
        self.points = torch.empty((0, 3), dtype=torch.float32, device=device)
        self.keys = torch.empty((0,), dtype=torch.int64, device=device)
        self.last_seen = torch.empty((0,), dtype=torch.int64, device=device)      # cycle index of the last observation
        self.empty_count = torch.empty((0,), dtype=torch.int16, device=device)   # consecutive cycles judged empty
        self.stale_count = torch.empty((0,), dtype=torch.int16, device=device)   # consecutive cycles seen but not refreshed
        self.unknown_count = torch.empty((0,), dtype=torch.int16, device=device) # consecutive cycles without any evidence
        self._dead: torch.Tensor | None = None                                    # rows to drop at the next update (forget())
        self.cycle = 0
        # Device counters (read with summary(); no per-cycle host sync): inserted, refreshed, kept occluded/unknown,
        # deleted (empty), capped (dropped for size), dropped as stale (seen but never refreshed), forgotten (forget()),
        # dropped as unknown (no evidence for too long)
        self._counters = torch.zeros(8, dtype=torch.int64, device=device)
        self._cam_cache: dict[int, tuple[bytes, torch.Tensor, list[float]]] = {}
        self.use_kernel = static_memory_kernel.available(device) and not bool(getattr(config, "reference_path", False))

    # ------------------------------------------------------------------ public
    @property
    def size(self) -> int:
        return int(self.points.shape[0])

    def forget(self, rows: torch.Tensor) -> None:
        """Drop these rows of the memory (indices into the cloud returned by the last update) at the next update.

        Used by the consumer for points it can prove are not obstacles, e.g. remembered points inside the robot's own
        bodies: robot pixels the self-filter missed enter the memory as rest points, and since the robot then stands in
        front of them they would never be seen as empty."""
        if rows.numel() == 0:
            return
        if self._dead is None or self._dead.shape[0] != self.points.shape[0]:
            self._dead = torch.zeros(self.points.shape[0], dtype=torch.bool, device=self.device)
        self._dead[rows.to(self.device)] = True

    def reset(self) -> None:
        self._dead = None
        self.points = self.points[:0]
        self.keys = self.keys[:0]
        self.last_seen = self.last_seen[:0]
        self.empty_count = self.empty_count[:0]
        self.stale_count = self.stale_count[:0]
        self.unknown_count = self.unknown_count[:0]

    def update(self, current: torch.Tensor, views: Sequence[tuple], upload: Callable[[tuple, Any], torch.Tensor] | None = None,
               current_keys: torch.Tensor | None = None, masks: Sequence[Any] | None = None) -> torch.Tensor:
        """Merge the rest points of this cycle into the memory and return the memory (the new rest scene).

        current: [N, 3] float32 world points, one per voxel (output of the rest-scene dedup).
        current_keys: their voxel keys, SORTED ascending (as the dedup produces them); computed here when None.
        views: (depth [H, W] float32 metres (host array or device tensor; 0 = invalid), (fx, fy, cx, cy),
               world_from_camera 4x4, excluded mask or None) - the rest-scene view tuples of this cycle.
        upload: callable (slot, array) -> device tensor for host depth images (the assembler's pinned upload).
        masks: per view, the occluder mask (bool [H, W] host or device, robot pixels) or None.

        The memory is kept sorted by voxel key, so the merge is two searchsorted passes and gathers: no sort of the
        union, no host round trip except the two sizes needed to slice the result."""
        self.cycle += 1
        current = current.to(self.device, torch.float32)
        if self._workspace is not None and current.shape[0] > 0:
            lo, hi = self._workspace
            inside = ((current >= lo) & (current <= hi)).all(dim=1)
            current = current[inside]
            current_keys = current_keys[inside] if current_keys is not None else None
        if current_keys is None:
            cur_keys = self._voxel_keys(current)
            cur_keys, order = torch.sort(cur_keys)
            current = current[order]
            n_valid = int((cur_keys != rest_scene_kernel.INVALID_KEY).sum())   # invalid keys sort last
            current, cur_keys = current[:n_valid], cur_keys[:n_valid]
        else:
            cur_keys = current_keys.to(self.device)
        n_cur = int(current.shape[0])
        m = int(self.points.shape[0])

        if m == 0:
            self.points, self.keys = current.clone(), cur_keys.clone()
            self.last_seen = torch.full((n_cur,), self.cycle, dtype=torch.int64, device=self.device)
            self.empty_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self.stale_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self.unknown_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self._counters[0] += n_cur
            return self.points

        # 1. Which memory voxels are observed again this cycle (both key lists sorted and unique).
        pos = torch.searchsorted(self.keys, cur_keys)                      # [N]
        pos_c = pos.clamp_(max=m - 1)
        found = self.keys[pos_c] == cur_keys                                # current point whose voxel is remembered
        refreshed = torch.zeros(m, dtype=torch.bool, device=self.device)
        refreshed[pos_c[found]] = True

        # 2. Visibility of every remembered point; delete the confirmed-empty and the stale noise voxels.
        empty_now, seen_now, known_now = self._observe(views, upload, masks)
        self.empty_count = torch.where(empty_now, self.empty_count + 1, torch.zeros_like(self.empty_count))
        keep = self.empty_count < int(self.cfg.empty_frames_to_delete)
        self._counters[3] += m - keep.sum()
        unknown_limit = int(self.cfg.unknown_frames_to_delete)
        if unknown_limit > 0:
            self.unknown_count = torch.where(known_now | refreshed, torch.zeros_like(self.unknown_count), self.unknown_count + 1)
            unknown = self.unknown_count >= unknown_limit
            self._counters[7] += (keep & unknown).sum()
            keep &= ~unknown
        if self._dead is not None and self._dead.shape[0] == m:
            self._counters[6] += (keep & self._dead).sum()
            keep &= ~self._dead
            dead_found = self._dead[pos_c[found]]
        else:
            dead_found = torch.zeros(pos_c[found].shape[0], dtype=torch.bool, device=self.device)
        self._dead = None
        stale_limit = int(self.cfg.seen_unrefreshed_frames_to_delete)
        if stale_limit > 0:
            self.stale_count = torch.where(seen_now & ~refreshed, self.stale_count + 1, torch.zeros_like(self.stale_count))
            stale = self.stale_count >= stale_limit
            self._counters[5] += (keep & stale).sum()
            keep &= ~stale
        self._counters[2] += (keep & ~seen_now & ~refreshed).sum()

        # 3. Refresh in place (position of the current observation, counters reset), then compact the survivors.
        idx_found = pos_c[found]
        self.points[idx_found] = current[found]
        self.last_seen[idx_found] = self.cycle
        self.empty_count[idx_found] = 0
        self.stale_count[idx_found] = 0
        self.unknown_count[idx_found] = 0
        keep[idx_found] = keep[idx_found] | ~dead_found                     # observed voxels are kept unless forgotten
        self._counters[1] += found.sum()
        surv = keep.nonzero(as_tuple=True)[0]                               # one host sync (size of the survivors)
        s_keys = self.keys[surv]
        s_points, s_seen = self.points[surv], self.last_seen[surv]
        s_empty, s_stale, s_unknown = self.empty_count[surv], self.stale_count[surv], self.unknown_count[surv]

        # 4. Insert the new voxels by a merge of two sorted key lists (no sort of the union).
        new = (~found).nonzero(as_tuple=True)[0]                            # second host sync (number of new voxels)
        k = int(new.shape[0])
        ms = int(s_keys.shape[0])
        self._counters[0] += k
        if k == 0:
            self.points, self.keys, self.last_seen = s_points, s_keys, s_seen
            self.empty_count, self.stale_count, self.unknown_count = s_empty, s_stale, s_unknown
        else:
            n_keys = cur_keys[new]
            ins = torch.searchsorted(s_keys, n_keys)                        # [k], non-decreasing
            ar_m = torch.arange(ms, device=self.device)
            ar_k = torch.arange(k, device=self.device)
            dst_m = ar_m + torch.searchsorted(ins, ar_m, right=True)        # new keys inserted before each survivor
            dst_k = ins + ar_k
            total = ms + k
            keys = torch.empty(total, dtype=torch.int64, device=self.device)
            points = torch.empty((total, 3), dtype=torch.float32, device=self.device)
            seen = torch.empty(total, dtype=torch.int64, device=self.device)
            empty = torch.zeros(total, dtype=torch.int16, device=self.device)
            stale = torch.zeros(total, dtype=torch.int16, device=self.device)
            unknown_c = torch.zeros(total, dtype=torch.int16, device=self.device)
            keys[dst_m], keys[dst_k] = s_keys, n_keys
            points[dst_m], points[dst_k] = s_points, current[new]
            seen[dst_m] = s_seen
            seen[dst_k] = self.cycle
            empty[dst_m], stale[dst_m], unknown_c[dst_m] = s_empty, s_stale, s_unknown
            self.points, self.keys, self.last_seen, self.empty_count, self.stale_count = points, keys, seen, empty, stale
            self.unknown_count = unknown_c

        cap = int(self.cfg.max_points)
        if cap > 0 and self.points.shape[0] > cap:              # drop the longest unseen points (keeps the key order)
            keep_idx = torch.topk(self.last_seen, cap, largest=True).indices.sort().values
            self._counters[4] += int(self.points.shape[0]) - cap
            self.points, self.keys = self.points[keep_idx], self.keys[keep_idx]
            self.last_seen, self.empty_count = self.last_seen[keep_idx], self.empty_count[keep_idx]
            self.stale_count, self.unknown_count = self.stale_count[keep_idx], self.unknown_count[keep_idx]
        return self.points

    def summary(self) -> str:
        c = [int(v) for v in self._counters.tolist()]
        return (f"static memory: {self.size} points after {self.cycle} cycles; inserted {c[0]}, refreshed {c[1]}, "
                f"kept while occluded/unknown {c[2]}, deleted as empty {c[3]}, dropped as stale {c[5]}, forgotten {c[6]}, "
                f"dropped as unknown {c[7]}, dropped for size {c[4]}")

    # ------------------------------------------------------------------ internals
    def _voxel_keys(self, points: torch.Tensor) -> torch.Tensor:
        if points.shape[0] == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.device)
        shifted = torch.floor((points - self._origin) * self._inv_voxel).to(torch.int64) + KEY_BIAS
        in_range = (shifted >= 0).all(dim=1) & (shifted <= KEY_MASK).all(dim=1)
        keys = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
        keys.masked_fill_(~in_range, rest_scene_kernel.INVALID_KEY)
        return keys

    def _margin(self, z: torch.Tensor) -> torch.Tensor:
        sigma = float(self.cfg.depth_sigma_a_m) + float(self.cfg.depth_sigma_b_per_m) * z * z
        return torch.clamp(float(self.cfg.margin_sigmas) * sigma, min=self._min_margin)

    def _camera(self, view: int, world_from_camera, intr) -> tuple[torch.Tensor, tuple[float, float, float, float]]:
        """Device camera<-world transform (3x4) of a view, re-uploaded only when the pose changes (static cameras
        never after the first cycle: a pageable host->device copy would synchronise the stream every cycle)."""
        T = np.eye(4, dtype=np.float64)
        wc = np.asarray(world_from_camera, dtype=np.float64)
        T[: wc.shape[0], :4] = wc[:, :4]
        key = T.tobytes()
        cached = self._cam_cache.get(view)
        if cached is None or cached[0] != key:
            inv = np.linalg.inv(T)[:3, :4]
            cam_from_world = torch.as_tensor(inv, dtype=torch.float32, device=self.device)
            cached = (key, cam_from_world, [float(v) for v in inv.reshape(-1)])
            self._cam_cache[view] = cached
        return cached[1], tuple(float(v) for v in intr)

    def _observe(self, views: Sequence[tuple], upload, masks: Sequence[Any] | None = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Per remembered point: (empty by any camera and seen by none, seen by any camera, any evidence at all)."""
        m = self.points.shape[0]
        min_z = max(self._min_depth, 1e-3)
        k = 2 * self._half + 1
        if self.use_kernel:
            seen8 = torch.zeros(m, dtype=torch.int8, device=self.device)
            empty8 = torch.zeros(m, dtype=torch.int8, device=self.device)
            known8 = torch.zeros(m, dtype=torch.int8, device=self.device)
        else:
            empty_any = torch.zeros(m, dtype=torch.bool, device=self.device)
            seen_any = torch.zeros(m, dtype=torch.bool, device=self.device)
            known_any = torch.zeros(m, dtype=torch.bool, device=self.device)
        for view, (depth, intr, world_from_camera, _excluded) in enumerate(views):
            mask = masks[view] if masks is not None and view < len(masks) else None
            if mask is not None:
                mask = mask.to(self.device) if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask), device=self.device)
                mask = (mask != 0) if mask.dtype != torch.bool else mask
                if tuple(mask.shape) != tuple(depth.shape):
                    mask = None
            if torch.is_tensor(depth):
                depth = depth.to(self.device)
            elif upload is not None:
                depth = upload(("static_memory_depth", view), np.asarray(depth, dtype=np.float32))
            else:
                depth = torch.as_tensor(np.asarray(depth, dtype=np.float32), device=self.device)
            if depth.ndim != 2 or depth.numel() == 0:
                continue
            height, width = int(depth.shape[0]), int(depth.shape[1])
            cam_from_world, (fx, fy, cx, cy) = self._camera(view, world_from_camera, intr)
            if self.use_kernel:
                static_memory_kernel.observe_view(
                    self.points, depth.contiguous(), self._cam_cache[view][2], (fx, fy, cx, cy),
                    (self._min_depth, self._max_depth), min_z,
                    (float(self.cfg.depth_sigma_a_m), float(self.cfg.depth_sigma_b_per_m), float(self.cfg.margin_sigmas), self._min_margin),
                    self._half, seen8, empty8, known8, mask.contiguous().to(torch.int8) if mask is not None else None,
                )
                continue
            p = torch.addmm(cam_from_world[:, 3], self.points, cam_from_world[:, :3].T)   # [M, 3] camera frame
            z = p[:, 2]
            zs = z.clamp(min=min_z)                      # projection of points behind the camera is masked below
            inv_z = 1.0 / zs
            u = torch.round(torch.addcmul(torch.full_like(z, cx), p[:, 0], inv_z, value=fx)).to(torch.int64)
            v = torch.round(torch.addcmul(torch.full_like(z, cy), p[:, 1], inv_z, value=fy)).to(torch.int64)
            in_img = (z > min_z) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
            flat = (v.clamp_(0, height - 1) * width).add_(u.clamp_(0, width - 1))
            valid = (depth >= self._min_depth) & (depth <= self._max_depth)      # finite by construction (0 = invalid)
            d_inf = torch.where(valid, depth, torch.full_like(depth, float("inf")))
            d_min = -F.max_pool2d(-d_inf[None, None], k, stride=1, padding=self._half)[0, 0] if self._half > 0 else d_inf
            d_c = depth.reshape(-1).index_select(0, flat)
            d_n = d_min.reshape(-1).index_select(0, flat)
            margin = self._margin(zs)
            c_valid = valid.reshape(-1).index_select(0, flat)
            seen = in_img & c_valid & ((d_c - zs).abs() <= margin)
            empty = in_img & (d_n > zs + margin) & torch.isfinite(d_n)
            occluded = in_img & c_valid & (d_c < zs - margin)
            if mask is not None:
                occluded |= in_img & mask.reshape(-1).index_select(0, flat)
            seen_any |= seen
            empty_any |= empty
            known_any |= seen | empty | occluded
        if self.use_kernel:
            seen_any = seen8 != 0
            return (empty8 != 0) & ~seen_any, seen_any, known8 != 0
        return empty_any & ~seen_any, seen_any, known_any
