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
        empty     all valid neighbours have d > z + margin (the ray passed through the point) and the pixel itself
                  is not a robot pixel -> delete after `empty_frames_to_delete` consecutive cycles. A depth beyond the
                  rest scene's range counts (a far wall; a pixel the camera reported as TOO FAR, which the assembler
                  hands over as FAR_DEPTH_M): out of range is not out of evidence
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

Young points beside the robot: a point the consumer reports as close to the robot (update(near_keys=...), from the
safety filter's SDF pass) whose voxel was observed in fewer than `min_observations_to_hold` cycles is not held - it is dropped
the first cycle it is not observed (also when a ray still ends near it: beside the robot that is the robot's surface). Robot pixels the self-filter missed for a frame (or a whole frame that passed
unfiltered) lie on the robot, the robot hides them the next cycle, and "behind a robot pixel" keeps a point without a time
limit: on the real arm 30 such points stood 1 cm beside the hand for as long as it did not move, and the safety filter
pushed away from them. A surface the robot approaches was observed long before the robot came close, and anything
that stays visible beside the robot is old enough after a fraction of a second. Away from the robot there is no such rule.

Robot-born points need an anchor: what the self-filter leaves of the robot is mostly not ON the robot but at the edge of
its silhouette in the image - mixed pixels between the arm and the background, lying anywhere along the ray behind the
arm (measured on the real installation: 63 % of the leftover points are within 4 pixels of a robot pixel in the camera
that measured them, against 0.5 % of the points of the scene; only a quarter are within 8 cm of a collision body). The
arm then hides them (and 82 % are within 16 pixels, against 6 %); the "behind a robot pixel" keeps a point without a time limit, and the leftovers of a moving arm
pile up into clusters. A point is therefore tagged ROBOT_BORN when it is inserted within `silhouette_px` of a robot pixel
(in a camera that measured it) or where the consumer reports the voxel close to the robot. Such a point is held only
once it is ANCHORED: connected, through neighbouring voxels, to a point that is not robot-born - the edge of a table the
arm uncovers is, a speck in free space is not. Without an anchor it lives while it is observed and is dropped the first
cycle it is not, whatever its age and however large its cluster has grown (`robot_born_needs_anchor`).

Two halves of a cycle: held(current keys) gives the remembered points to add to this cycle's rest points - those whose
voxel is not observed now - with one binary search; that is all the latency-critical path pays. update() (merge, visibility
of every remembered point, deletions) runs after the consumer's command is out (SceneCloudAssembler.finish_memory), so a
deletion shows in the cloud one cycle later. Seen-but-not-observed points (depth-noise voxels beside a surface that is in
the current cloud) stay inside the memory and are not handed out.

Evidence mode (`evidence: true`): how often a point has been observed decides whether it may be held at all. Without it
every point that was in the cloud for one cycle is held like a table: on the real installation 77 % of the points held
for more than a second had been observed in fewer than 5 cycles - what the moving arm, a hand, an unmasked object leave
behind where one camera is blocked and the other has no depth - and stood for tens of seconds. In evidence mode

    * a point is CONFIRMED once its voxel was observed in `confirm_observations` cycles (ten = a third of a second at
      rest; what moves faster than a voxel per that time never gets there). Only confirmed points are held without a
      time limit; an unconfirmed one is handed out for `candidate_hold_frames` cycles after its last observation and then
      dropped, whatever hides it;
    * nothing lowers the confidence of a hidden point: behind the robot or another object there is no evidence, and the
      count stays. Only a ray through the point's place removes it, as before (`empty_frames_to_delete`);
    * the rules for the robot's leftovers (young points, robot-born points and their anchors) are not applied: on a
      three-minute recording of the real installation (replayed through every variant) they changed nothing once ten
      observations were required - floating held cells 121 with them off against 113 with them on, 549 without evidence
      mode; static patches kept through an occlusion 50 % / 53 % / 57 % - and they cost half of the update.

Semi-static scenes follow from the same rule: an object that is put down is in the cloud at once (it is observed) and is
remembered a third of a second later; one that is taken away is released as soon as a camera sees through its place,
however long it stood there; taken away while hidden, its place stays blocked until a camera sees it again.

Size: the memory holds what the pipeline's zone (zone.py) lets through. With a whole room in view and no zone it runs
into max_points within a second, and the cap drops the longest-unseen points - the hidden ones the memory exists for.
The visibility test reads the depth images BEFORE the zone blanked them: a ray through a remembered point usually ends
on a surface outside the zone.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from . import rest_scene_kernel, static_memory_kernel

_NO_MAX = 1.0e30    # the visibility test has no far limit (depth beyond the range = the ray passed)
KEY_BIAS = 1 << 20
ROBOT_BORN = 1      # tags: first observed close to the robot (consumer's near_keys) or at the robot's silhouette in the image
ANCHORED = 2        #       ... and connected to something that was not
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
        self.obs_count = torch.empty((0,), dtype=torch.int16, device=device)     # cycles in which the voxel was observed (saturating)
        self.near_flag = torch.empty((0,), dtype=torch.bool, device=device)      # reported close to the robot while young
        self.tags = torch.empty((0,), dtype=torch.uint8, device=device)          # ROBOT_BORN | ANCHORED (see module text)
        offs = [(dx << 42) + (dy << 21) + dz for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]
        self._neighbours = torch.tensor(offs, dtype=torch.int64, device=device)
        self.cycle = 0
        # Device counters (read with summary(); no per-cycle host sync): inserted, refreshed, kept occluded/unknown,
        # deleted (empty), capped (dropped for size), dropped as stale (seen but never refreshed), forgotten (forget()),
        # dropped as unknown (no evidence for too long)
        # ... and dropped young (hidden before min_observations_to_hold observations)
        # ... and dropped as robot-born without an anchor
        # ... and dropped as unconfirmed (evidence mode: not observed for candidate_hold_frames before being confirmed)
        self._counters = torch.zeros(11, dtype=torch.int64, device=device)
        self.evidence = bool(getattr(config, "evidence", False))
        self._confirm_after = int(getattr(config, "confirm_observations", 10))
        self._candidate_hold = int(getattr(config, "candidate_hold_frames", 9))
        self._cam_cache: dict[int, tuple[bytes, torch.Tensor, list[float]]] = {}
        self.use_kernel = static_memory_kernel.available(device) and not bool(getattr(config, "reference_path", False))

    # ------------------------------------------------------------------ public
    @property
    def size(self) -> int:
        return int(self.points.shape[0])

    @staticmethod
    def _isin_sorted(keys: torch.Tensor, sorted_keys: torch.Tensor) -> torch.Tensor:
        """keys[i] in sorted_keys (ascending), by binary search."""
        if sorted_keys.shape[0] == 0 or keys.shape[0] == 0:
            return torch.zeros(keys.shape[0], dtype=torch.bool, device=keys.device)
        pos = torch.searchsorted(sorted_keys, keys).clamp_(max=sorted_keys.shape[0] - 1)
        return sorted_keys[pos] == keys

    def held(self, current_keys: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(points, voxel keys) of the remembered points to ADD to the rest points observed this cycle: those whose
        voxel is not among `current_keys` (sorted) - the hidden ones. This is all the latency-critical path does with
        the memory (one binary search over the memory); update() runs after the consumer's command is out.

        Left out: points seen but not observed at the last update (stale_count > 0) - depth-noise voxels beside a
        surface that IS in the current cloud; they are kept inside the memory for the stale rule only."""
        if self.points.shape[0] == 0:
            return self.points, self.keys
        mask = ~self._isin_sorted(self.keys, current_keys.to(self.device)) & (self.stale_count == 0)
        if not self.evidence and self.tags.shape[0] == mask.shape[0] and bool(getattr(self.cfg, "robot_born_needs_anchor", True)):
            # robot-born without an anchor: never held (the update drops it)
            mask &= (self.tags & (ROBOT_BORN | ANCHORED)) != ROBOT_BORN
        idx = mask.nonzero(as_tuple=True)[0]
        return self.points.index_select(0, idx), self.keys.index_select(0, idx)

    def reset(self) -> None:
        self.near_flag = self.near_flag[:0]
        self.tags = self.tags[:0]
        self.points = self.points[:0]
        self.keys = self.keys[:0]
        self.last_seen = self.last_seen[:0]
        self.empty_count = self.empty_count[:0]
        self.stale_count = self.stale_count[:0]
        self.unknown_count = self.unknown_count[:0]
        self.obs_count = self.obs_count[:0]

    def update(self, current: torch.Tensor, views: Sequence[tuple], upload: Callable[[tuple, Any], torch.Tensor] | None = None,
               current_keys: torch.Tensor | None = None, masks: Sequence[Any] | None = None,
               forget_keys: torch.Tensor | None = None, near_keys: torch.Tensor | None = None) -> torch.Tensor:
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
        self._update(current, current_keys, views, upload, masks, forget_keys, near_keys)
        return self.points

    def sorted_keys(self, points: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(points, voxel keys) sorted by key, points outside the key range dropped."""
        keys = self._voxel_keys(points)
        keys, order = torch.sort(keys)
        points = points[order]
        n_valid = int((keys != rest_scene_kernel.INVALID_KEY).sum())           # invalid keys sort last
        return points[:n_valid], keys[:n_valid]

    def _update(self, current, current_keys, views, upload, masks, forget_keys=None, near_keys=None) -> None:
        if current_keys is None:
            current, cur_keys = self.sorted_keys(current)
        else:
            cur_keys = current_keys.to(self.device)
        # Consumer feedback of this cycle, by voxel key: forget_keys = not obstacles (inside the robot): dropped and not
        # taken from this cycle's observation either; near_keys = close to the robot (see the young rule below).
        if forget_keys is not None and forget_keys.numel() > 0:
            forget_keys = torch.sort(forget_keys.to(self.device)).values
            observed = ~self._isin_sorted(cur_keys, forget_keys)
            current, cur_keys = current[observed], cur_keys[observed]
        else:
            forget_keys = None
        if near_keys is not None and near_keys.numel() > 0 and not self.evidence:
            near_keys = torch.sort(near_keys.to(self.device)).values
        else:
            near_keys = None                                  # evidence mode: the young / robot-born rules are not applied
        n_cur = int(current.shape[0])
        m = int(self.points.shape[0])

        if m == 0:
            self.points, self.keys = current.clone(), cur_keys.clone()
            self.last_seen = torch.full((n_cur,), self.cycle, dtype=torch.int64, device=self.device)
            self.empty_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self.stale_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self.unknown_count = torch.zeros(n_cur, dtype=torch.int16, device=self.device)
            self.obs_count = torch.ones(n_cur, dtype=torch.int16, device=self.device)
            self.near_flag = (self._isin_sorted(cur_keys, near_keys) if near_keys is not None
                              else torch.zeros(n_cur, dtype=torch.bool, device=self.device))
            if self.evidence:
                self.tags = torch.zeros(n_cur, dtype=torch.uint8, device=self.device)
            else:
                self.tags = self._born_tags(current, cur_keys, near_keys, views, upload, masks)
                self._anchor()
            self._counters[0] += n_cur
            return

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
        dead = None
        if forget_keys is not None:
            dead = self._isin_sorted(self.keys, forget_keys)
            self._counters[6] += (keep & dead).sum()
            keep &= ~dead
        if self.near_flag.shape[0] != m:
            self.near_flag = torch.zeros(m, dtype=torch.bool, device=self.device)
        if near_keys is not None:
            self.near_flag = self.near_flag | self._isin_sorted(self.keys, near_keys)
        stale_limit = int(self.cfg.seen_unrefreshed_frames_to_delete)
        if stale_limit > 0:
            self.stale_count = torch.where(seen_now & ~refreshed, self.stale_count + 1, torch.zeros_like(self.stale_count))
            stale = self.stale_count >= stale_limit
            self._counters[5] += (keep & stale).sum()
            keep &= ~stale
        hold_after = 0 if self.evidence else int(getattr(self.cfg, "min_observations_to_hold", 0))
        if self.tags.shape[0] != m:
            self.tags = torch.zeros(m, dtype=torch.uint8, device=self.device)
        if not self.evidence and bool(getattr(self.cfg, "robot_born_needs_anchor", True)):
            # robot-born and not connected to anything else: lives only while it is observed
            loose = ((self.tags & (ROBOT_BORN | ANCHORED)) == ROBOT_BORN) & ~refreshed
            self._counters[9] += (keep & loose).sum()
            keep &= ~loose
        young = None
        if hold_after > 1:
            young = (self.obs_count < hold_after) & ~refreshed & self.near_flag
            self._counters[8] += (keep & young).sum()
            keep &= ~young
            self.near_flag = self.near_flag & (self.obs_count < hold_after)     # old enough: the flag has done its work
        if self.evidence:
            # unconfirmed and not observed for longer than the hold: dropped, whatever hides it
            late = (self.obs_count < self._confirm_after) & ~refreshed & ((self.cycle - self.last_seen) > self._candidate_hold)
            self._counters[10] += (keep & late).sum()
            keep &= ~late
        self._counters[2] += (keep & ~seen_now & ~refreshed).sum()

        # 3. Refresh in place (position of the current observation, counters reset), then compact the survivors.
        idx_found = pos_c[found]
        self.points[idx_found] = current[found]
        self.last_seen[idx_found] = self.cycle
        self.empty_count[idx_found] = 0
        self.stale_count[idx_found] = 0
        self.unknown_count[idx_found] = 0
        self.obs_count[idx_found] = (self.obs_count[idx_found] + 1).clamp_(max=30000)
        keep[idx_found] = True                                              # observed voxels are kept (forgotten ones are not among them)
        self._counters[1] += found.sum()
        surv = keep.nonzero(as_tuple=True)[0]                               # one host sync (size of the survivors)
        s_keys = self.keys[surv]
        s_points, s_seen = self.points[surv], self.last_seen[surv]
        s_empty, s_stale, s_unknown = self.empty_count[surv], self.stale_count[surv], self.unknown_count[surv]
        s_obs, s_near, s_tags = self.obs_count[surv], self.near_flag[surv], self.tags[surv]

        # 4. Insert the new voxels by a merge of two sorted key lists (no sort of the union).
        new = (~found).nonzero(as_tuple=True)[0]                            # second host sync (number of new voxels)
        k = int(new.shape[0])
        ms = int(s_keys.shape[0])
        self._counters[0] += k
        if k == 0:
            self.points, self.keys, self.last_seen = s_points, s_keys, s_seen
            self.empty_count, self.stale_count, self.unknown_count = s_empty, s_stale, s_unknown
            self.obs_count, self.near_flag, self.tags = s_obs, s_near, s_tags
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
            obs = torch.ones(total, dtype=torch.int16, device=self.device)
            obs[dst_m] = s_obs
            self.obs_count = obs
            near = torch.zeros(total, dtype=torch.bool, device=self.device)
            near[dst_m] = s_near
            if near_keys is not None:
                near[dst_k] = self._isin_sorted(n_keys, near_keys)
            self.near_flag = near
            tags = torch.zeros(total, dtype=torch.uint8, device=self.device)
            tags[dst_m] = s_tags
            if not self.evidence:
                tags[dst_k] = self._born_tags(current[new], n_keys, near_keys, views, upload, masks)
            self.tags = tags
        if not self.evidence:
            self._anchor()

        cap = int(self.cfg.max_points)
        if cap > 0 and self.points.shape[0] > cap:              # drop the longest unseen points (keeps the key order)
            keep_idx = torch.topk(self.last_seen, cap, largest=True).indices.sort().values
            self._counters[4] += int(self.points.shape[0]) - cap
            self.points, self.keys = self.points[keep_idx], self.keys[keep_idx]
            self.last_seen, self.empty_count = self.last_seen[keep_idx], self.empty_count[keep_idx]
            self.stale_count, self.unknown_count = self.stale_count[keep_idx], self.unknown_count[keep_idx]
            self.obs_count, self.near_flag = self.obs_count[keep_idx], self.near_flag[keep_idx]
            self.tags = self.tags[keep_idx]

    def summary(self) -> str:
        c = [int(v) for v in self._counters.tolist()]
        return (f"static memory: {self.size} points after {self.cycle} cycles; inserted {c[0]}, refreshed {c[1]}, "
                f"kept while occluded/unknown {c[2]}, deleted as empty {c[3]}, dropped as stale {c[5]}, forgotten {c[6]}, "
                f"dropped as unknown {c[7]}, dropped young {c[8]}, dropped robot-born without anchor {c[9]}, dropped for size {c[4]}"
                + self._evidence_summary(int(c[10])))

    # ------------------------------------------------------------------ internals
    def _born_tags(self, points, keys, near_keys, views, upload, masks) -> torch.Tensor:
        """Tag of newly inserted points: ROBOT_BORN when the consumer reports the voxel close to the robot (near_keys) or
        when the camera that measured the point has a robot pixel within silhouette_px of it."""
        born = self._isin_sorted(keys, near_keys) if near_keys is not None else torch.zeros(keys.shape[0], dtype=torch.bool, device=self.device)
        radius = int(getattr(self.cfg, "silhouette_px", 0))
        if radius > 0 and masks is not None and points.shape[0] > 0:
            for view, (depth, intr, world_from_camera, _excluded) in enumerate(views):
                mask = masks[view] if view < len(masks) else None
                if mask is None:
                    continue
                mask = mask.to(self.device) if torch.is_tensor(mask) else torch.as_tensor(np.asarray(mask), device=self.device)
                if torch.is_tensor(depth):
                    depth = depth.to(self.device)
                elif upload is not None:
                    depth = upload(("static_memory_depth", view), np.asarray(depth, dtype=np.float32))
                else:
                    depth = torch.as_tensor(np.asarray(depth, dtype=np.float32), device=self.device)
                if depth.ndim != 2 or tuple(mask.shape) != tuple(depth.shape):
                    continue
                height, width = int(depth.shape[0]), int(depth.shape[1])
                cam_from_world, (fx, fy, cx, cy) = self._camera(view, world_from_camera, intr)
                p = points @ cam_from_world[:, :3].T + cam_from_world[:, 3]
                z = p[:, 2].clamp(min=1e-3)
                u = torch.round(fx * p[:, 0] / z + cx).to(torch.int64)
                v = torch.round(fy * p[:, 1] / z + cy).to(torch.int64)
                inside = (p[:, 2] > 1e-3) & (u >= 0) & (u < width) & (v >= 0) & (v < height)
                flat = v.clamp_(0, height - 1) * width + u.clamp_(0, width - 1)
                measured = inside & ((depth.reshape(-1).index_select(0, flat) - z).abs() <= self._margin(z))
                wide = F.max_pool2d((mask != 0).to(torch.float32)[None, None], 2 * radius + 1, stride=1, padding=radius)[0, 0]
                born |= measured & (wide.reshape(-1).index_select(0, flat) > 0)
        return born.to(torch.uint8) * ROBOT_BORN

    def _anchor(self, rounds: int = 3) -> None:
        """Robot-born points next to (26 neighbours on the lattice) a point that is not robot-born, or that is anchored
        itself, become anchored. A few rounds per cycle; the flag stays, so chains settle over the following cycles."""
        m = int(self.keys.shape[0])
        if m == 0 or self.tags.shape[0] != m:
            return
        idx = ((self.tags & (ROBOT_BORN | ANCHORED)) == ROBOT_BORN).nonzero(as_tuple=True)[0]
        for _ in range(rounds):
            if idx.numel() == 0:
                return
            cand = self.keys[idx][:, None] + self._neighbours[None]
            pos = torch.searchsorted(self.keys, cand.reshape(-1)).clamp_(max=m - 1).reshape(cand.shape)
            solid = (self.keys[pos] == cand) & ((self.tags[pos] & (ROBOT_BORN | ANCHORED)) != ROBOT_BORN)
            hit = solid.any(dim=1)
            self.tags[idx[hit]] |= ANCHORED
            if not bool(hit.any()):
                return
            idx = idx[~hit]

    def _evidence_summary(self, dropped_unconfirmed: int) -> str:
        if not self.evidence:
            return ""
        m = int(self.obs_count.shape[0])
        confirmed = int((self.obs_count >= self._confirm_after).sum()) if m else 0
        return f"; evidence: confirmed {confirmed} of {m}, dropped unconfirmed {dropped_unconfirmed}"

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
                    (self._min_depth, _NO_MAX), min_z,
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
            # 0 = no measurement. A depth beyond the range of the rest scene (a far wall, or the assembler's FAR_DEPTH_M at
            # a pixel the camera reported as too far) is a measurement all the same: the ray passed everything nearer.
            valid = depth >= self._min_depth
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
                robot = in_img & mask.reshape(-1).index_select(0, flat)
                occluded |= robot
                empty &= ~robot                          # behind a robot pixel: the rays beside it say nothing about the point
            seen_any |= seen
            empty_any |= empty
            known_any |= seen | empty | occluded
        if self.use_kernel:
            seen_any = seen8 != 0
            return (empty8 != 0) & ~seen_any, seen_any, known8 != 0
        return empty_any & ~seen_any, seen_any, known_any
