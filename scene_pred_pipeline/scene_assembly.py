"""Assemble the merged full-scene cloud at the end of every pipeline cycle.

Output (all CUDA tensors, world frame), written into SceneVelocityOutput:
    scene_points     [N, 3] float32  meters
    scene_velocity   [N, 3] float32  m/s
    scene_track_ids  [N]    int32
    scene_num_dynamic        rows [0, n) are tracked-instance points

Dynamic part: DifFlow/recovery points and velocities when the flow pair was
valid, otherwise the current tracked points with zero velocity (so tracked
objects never disappear from the cloud).
Rest part: dense depth backprojection of every camera minus the tracked and
excluded masks, voxel-deduplicated on one world lattice (the tracker's shared
lattice, or scene_cloud.voxel_size_m), with zero velocity and track id 0.
Static-only mode (no tracked classes): the rest part is built directly from the
synchronized RGB-D frame and is the whole cloud.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from . import rest_scene_kernel
from .config import PipelineConfig
from .data_types import MultiCameraFrame, SceneVelocityOutput


class SceneCloudAssembler:
    """Builds the merged scene cloud from tracker results (masks excluded) or, without a
    tracker (static-only mode), directly from the synchronized RGB-D frame."""

    def __init__(self, config: PipelineConfig, *, tracker_config: Any | None = None) -> None:
        self.config = config
        self.scene_cfg = config.scene_cloud
        self.device = torch.device(config.runtime.device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self._ray_cache: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]] = {}
        # Pinned host staging + device buffer per upload slot (depth, mask, transform of each view): a pageable
        # host->device copy synchronizes the stream (the CPU waits for all queued DifFlow work); a pinned
        # non_blocking copy does not. The event guards the host buffer until its previous copy completed.
        self._uploads: dict[tuple[Any, ...], tuple[torch.Tensor, torch.Tensor, torch.cuda.Event]] = {}
        self._fused_buffers: tuple[torch.Tensor, torch.Tensor] | None = None   # points, keys of the fused path
        self._origin: torch.Tensor | None = None
        self._inv_voxel_size = 0.0
        self.last_rest_input = 0   # valid depth pixels of all views in the last rest-scene build (before dedup)
        self._min_depth_m = 0.0
        self._max_depth_m = float("inf")
        if self.scene_cfg.include_rest_points:
            self._configure_rest_scene(tracker_config)
        self._workspace = None
        if self.scene_cfg.workspace_min is not None and self.scene_cfg.workspace_max is not None:
            self._workspace = (
                torch.as_tensor(self.scene_cfg.workspace_min, dtype=torch.float32, device=self.device),
                torch.as_tensor(self.scene_cfg.workspace_max, dtype=torch.float32, device=self.device),
            )

    # ------------------------------------------------------------------ public
    def assemble(self, output: SceneVelocityOutput, frame: MultiCameraFrame | None = None) -> None:
        """Fill output.scene_* from the tracker results in `output.view_results`, or from
        `frame` (dense RGB-D) when there are no tracker results (static-only mode)."""
        dyn_region = None                    # velocity regions of the dynamic rows (when computed)
        if output.flow_valid and output.flow_points.shape[0] > 0:
            dyn_points = output.flow_points
            dyn_velocity = output.flow_velocity
            dyn_ids = output.flow_track_ids
            dyn_region = getattr(output, "flow_velocity_region", None)
            # Tracked instances without a flow estimate this cycle (an id seen for the first
            # time, e.g. after a cross-view split or a new detection) must not vanish from the
            # cloud: keep their points with zero velocity so the safety filter still sees them.
            tracked_ids = output.tracked_track_ids
            # Host check first: the device path below syncs (nonzero) and is needed only when some tracked
            # instance has no flow estimate (or the instance ids are unknown).
            maybe_missing = not output.tracked_instance_ids or bool(
                set(output.tracked_instance_ids) - set(output.common_track_ids)
            )
            if tracked_ids.shape[0] > 0 and output.common_track_ids and maybe_missing:
                common = torch.as_tensor(list(output.common_track_ids), dtype=tracked_ids.dtype, device=tracked_ids.device)
                missing = ~torch.isin(tracked_ids, common)
                if bool(missing.any()):
                    idx = missing.nonzero(as_tuple=True)[0]
                    extra_points = output.tracked_points.index_select(0, idx)
                    dyn_points = torch.cat((dyn_points.to(extra_points.device), extra_points), dim=0)
                    dyn_velocity = torch.cat((dyn_velocity.to(extra_points.device), torch.zeros_like(extra_points)), dim=0)
                    dyn_ids = torch.cat((dyn_ids.to(extra_points.device), tracked_ids.index_select(0, idx)), dim=0)
                    if dyn_region is not None:
                        dyn_region = torch.cat((dyn_region.to(extra_points.device), extra_points.new_zeros((idx.shape[0], 3, 3))), dim=0)
        else:
            dyn_points = output.tracked_points
            dyn_velocity = torch.zeros_like(dyn_points)
            dyn_ids = output.tracked_track_ids
        dyn_points = dyn_points.to(self.device, torch.float32)
        dyn_velocity = dyn_velocity.to(self.device, torch.float32)
        dyn_ids = dyn_ids.to(self.device, torch.int32)
        if dyn_region is not None:
            dyn_region = dyn_region.to(self.device, torch.float32)

        self.last_rest_input = 0
        if self.scene_cfg.include_rest_points:
            if output.view_results:
                rest = self.rest_scene_points(output)
            elif frame is not None:
                rest = self.rest_points_from_frame(frame)
            else:
                rest = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            rest = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        num_input = self.last_rest_input + int(dyn_points.shape[0])

        points = torch.cat((dyn_points, rest), dim=0)
        velocity = torch.cat((dyn_velocity, torch.zeros_like(rest)), dim=0)
        track_ids = torch.cat((dyn_ids, torch.zeros(rest.shape[0], dtype=torch.int32, device=self.device)), dim=0)
        num_dynamic = int(dyn_points.shape[0])

        if self._workspace is not None and points.shape[0] > 0:
            lo, hi = self._workspace
            keep = ((points >= lo) & (points <= hi)).all(dim=1)
            # Keep the dynamic-first ordering; count dynamic survivors.
            if dyn_region is not None:
                dyn_region = dyn_region.index_select(0, keep[:num_dynamic].nonzero(as_tuple=True)[0])
            num_dynamic = int(keep[:num_dynamic].sum().item())
            index = keep.nonzero(as_tuple=True)[0]
            points = points.index_select(0, index)
            velocity = velocity.index_select(0, index)
            track_ids = track_ids.index_select(0, index)

        output.scene_points = points.contiguous()
        output.scene_velocity = velocity.contiguous()
        output.scene_track_ids = track_ids.contiguous()
        output.scene_num_dynamic = num_dynamic
        output.scene_velocity_region = dyn_region.contiguous() if dyn_region is not None else None
        output.scene_num_input = num_input

    # ------------------------------------------------------------------ rest scene (moved from RosVisualizer)
    def _configure_rest_scene(self, tracker_config: Any | None) -> None:
        """Lattice and depth limits: scene_cloud.voxel_size_m overrides the tracker lattice;
        without a tracker (static-only) the scene_cloud values are used throughout."""
        origin_world = np.zeros(3, dtype=np.float32)
        voxel_size_m = self.scene_cfg.voxel_size_m
        min_depth_m, max_depth_m = float(self.scene_cfg.depth_min_m), float(self.scene_cfg.depth_max_m)
        if tracker_config is not None:
            try:
                if voxel_size_m is None:
                    voxel_size_m = float(tracker_config.shared_voxel_grid.voxel_size_m)
                origin_world = np.asarray(tracker_config.shared_voxel_grid.origin_world, dtype=np.float32).reshape(3)
                min_depth_m = float(tracker_config.postprocess.min_valid_depth_m)
                max_depth_m = float(tracker_config.postprocess.max_valid_depth_m)
            except (AttributeError, TypeError, ValueError) as exc:
                raise ValueError(
                    "Rest-scene assembly requires tracker shared_voxel_grid.voxel_size_m/"
                    "origin_world and postprocess min/max depth settings."
                ) from exc
        if voxel_size_m is None:
            raise ValueError("scene_cloud.voxel_size_m is required when no tracker config is available")
        voxel_size_m = float(voxel_size_m)
        if voxel_size_m <= 0.0:
            raise ValueError(f"rest-scene voxel size must be positive, got {voxel_size_m}")
        if max_depth_m < min_depth_m:
            raise ValueError("rest-scene max depth must be >= min depth")
        self.voxel_size_m = voxel_size_m
        self._inv_voxel_size = 1.0 / voxel_size_m
        self._origin = torch.as_tensor(origin_world, dtype=torch.float32, device=self.device).contiguous()
        self._origin_host = tuple(float(v) for v in np.asarray(origin_world, dtype=np.float32).reshape(3))
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m

    def _upload(self, slot: tuple[Any, ...], array: np.ndarray) -> torch.Tensor:
        """Device copy of a host array through a reused pinned buffer, without a stream sync.

        The returned device tensor is reused by the next upload to the same slot; every reader runs on the
        current stream before that upload is enqueued, so stream order protects it. Callers that keep the
        result beyond the cycle must clone it."""
        array = np.ascontiguousarray(array)
        if self.device.type != "cuda":
            return torch.as_tensor(array, device=self.device)
        key = (slot, array.shape, array.dtype.str)
        entry = self._uploads.get(key)
        if entry is None:
            host = torch.empty(array.shape, dtype=torch.from_numpy(array).dtype, pin_memory=True)
            entry = (host, torch.empty(array.shape, dtype=host.dtype, device=self.device), torch.cuda.Event())
            self._uploads[key] = entry
        else:
            entry[2].synchronize()                  # the previous copy out of `host` finished (one cycle ago)
        host, dev, done = entry
        host.numpy()[...] = array
        dev.copy_(host, non_blocking=True)
        done.record(torch.cuda.current_stream(self.device))
        return dev

    def _rays(self, height: int, width: int, fx: float, fy: float, cx: float, cy: float) -> tuple[torch.Tensor, torch.Tensor]:
        key = (int(height), int(width), float(fx), float(fy), float(cx), float(cy), str(self.device))
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        x_ray = (torch.arange(width, dtype=torch.float32, device=self.device) - float(cx)) / float(fx)
        y_ray = (torch.arange(height, dtype=torch.float32, device=self.device) - float(cy)) / float(fy)
        cached = (x_ray.repeat(height), y_ray.repeat_interleave(width))
        self._ray_cache[key] = cached
        return cached

    def _exclusion_mask_gpu(self, result: Any, shape: tuple[int, int], view: int) -> torch.Tensor | None:
        """Union of tracked final masks + the tracker-provided exclusion-only mask (None: nothing excluded).

        Host masks (rare) are OR-ed on the CPU and uploaded once; device masks are OR-ed on the GPU without a
        zero host array (the tracker's masks are device tensors)."""
        excluded_cpu: np.ndarray | None = None
        gpu_masks: list[torch.Tensor] = []
        for instance in getattr(result, "instances", ()):
            mask = getattr(instance, "mask", None)
            if mask is not None and not torch.is_tensor(mask):
                value = np.asarray(mask)
                if value.shape == shape:
                    if excluded_cpu is None:
                        excluded_cpu = np.zeros(shape, dtype=bool)
                    np.logical_or(excluded_cpu, value, out=excluded_cpu)
                    continue
            elif torch.is_tensor(mask) and tuple(mask.shape) == shape:
                gpu_masks.append(mask)
                continue
            mask_gpu = getattr(instance, "mask_gpu", None)
            if torch.is_tensor(mask_gpu) and tuple(mask_gpu.shape) == shape:
                gpu_masks.append(mask_gpu)

        semantic_exclusion = getattr(result, "exclusion_mask_gpu", None)
        if torch.is_tensor(semantic_exclusion) and tuple(semantic_exclusion.shape) == shape:
            gpu_masks.append(semantic_exclusion)
        elif semantic_exclusion is not None:
            value = np.asarray(semantic_exclusion)
            if value.shape == shape:
                if excluded_cpu is None:
                    excluded_cpu = np.zeros(shape, dtype=bool)
                np.logical_or(excluded_cpu, value, out=excluded_cpu)

        if excluded_cpu is not None:
            excluded = self._upload(("excluded", view), excluded_cpu).clone()
        elif gpu_masks:
            excluded = gpu_masks.pop(0).to(self.device) != 0       # a new bool tensor (also for bool masks)
        else:
            return None
        for mask_gpu in gpu_masks:
            if mask_gpu.device != self.device:
                mask_gpu = mask_gpu.to(self.device)
            excluded.logical_or_(mask_gpu if mask_gpu.dtype == torch.bool else mask_gpu != 0)
        return excluded

    def rest_scene_points(self, output: SceneVelocityOutput) -> torch.Tensor:
        """Rest scene from tracker view results: dense depth minus tracked/excluded masks."""
        views = []
        for result in output.view_results.values():
            frame = result.frame
            depth_cpu = np.asarray(frame.depth_m, dtype=np.float32)
            if depth_cpu.ndim != 2 or depth_cpu.size == 0:
                continue
            world_from_camera = getattr(frame, "world_from_camera", None)
            if world_from_camera is None:
                continue
            intr = frame.intrinsics
            excluded = self._exclusion_mask_gpu(result, tuple(depth_cpu.shape), len(views))
            views.append((depth_cpu, (float(intr.fx), float(intr.fy), float(intr.cx), float(intr.cy)), np.asarray(world_from_camera, dtype=np.float32), excluded))
        return self._rest_points_from_views(views)

    def rest_points_from_frame(self, frame: MultiCameraFrame) -> torch.Tensor:
        """Rest scene directly from the synchronized RGB-D bundle (no tracker: nothing excluded)."""
        views = []
        for cam in frame.cameras.values():
            depth_cpu = np.asarray(cam.depth, dtype=np.float32)
            if depth_cpu.ndim != 2 or depth_cpu.size == 0:
                continue
            K = np.asarray(cam.K, dtype=np.float64).reshape(3, 3)
            views.append((depth_cpu, (float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])), np.asarray(cam.T_world_camera, dtype=np.float32), None))
        return self._rest_points_from_views(views)

    def _rest_points_fused(self, views) -> torch.Tensor:
        """All views through the fused Triton kernel (rest_scene_kernel.py) into one buffer, then the voxel dedup."""
        sizes = [int(depth_cpu.size) for depth_cpu, *_ in views]
        total = sum(sizes)
        buffers = self._fused_buffers
        if buffers is None or buffers[0].shape[0] != total:
            buffers = (torch.empty((total, 3), dtype=torch.float32, device=self.device),
                       torch.empty((total,), dtype=torch.int64, device=self.device))
            self._fused_buffers = buffers
        points, keys = buffers
        start = 0
        for view, ((depth_cpu, (fx, fy, cx, cy), transform_cpu, excluded), size) in enumerate(zip(views, sizes)):
            depth = self._upload(("depth", view), depth_cpu).reshape(-1)
            excl = excluded.reshape(-1).to(self.device) if excluded is not None else None
            rest_scene_kernel.backproject_view(
                depth, excl, (fx, fy, cx, cy, int(depth_cpu.shape[1])), np.asarray(transform_cpu, dtype=np.float64).tolist(),
                self._origin_host, self._inv_voxel_size, (self._min_depth_m, self._max_depth_m),
                points[start:start + size], keys[start:start + size],
            )
            start += size
        return self._dedup(points, keys)

    def _dedup(self, points: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """One point per voxel key (the first in sort order); INVALID_KEY marks dropped pixels.

        Sets last_rest_input to the number of valid pixels; it is read after the boolean index, which already
        synchronised the stream, so the count costs no extra stall."""
        invalid_key = rest_scene_kernel.INVALID_KEY
        sorted_keys, order = torch.sort(keys)
        valid = sorted_keys != invalid_key
        keep = valid.clone()
        keep[1:] &= sorted_keys[1:] != sorted_keys[:-1]
        rest = points[order[keep]]
        self.last_rest_input = int(valid.sum())
        return rest

    def _rest_points_from_views(self, views) -> torch.Tensor:
        """Dense backprojection of every view, world transform, voxel dedup on one lattice; all on CUDA.

        views: iterable of (depth_m [H,W] float32 numpy, (fx, fy, cx, cy), world_from_camera 4x4/3x4, excluded mask [H,W] bool tensor or None)
        """
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        self.last_rest_input = 0
        if self._origin is None:
            return empty
        views = [v for v in views if v[2].shape in {(4, 4), (3, 4)}]
        if not views:
            return empty
        if rest_scene_kernel.available(self.device):
            return self._rest_points_fused(views)
        # Reference path (no Triton): same arithmetic in plain torch ops.
        all_points: list[torch.Tensor] = []
        all_keys: list[torch.Tensor] = []
        bias = 1 << 20
        key_mask = (1 << 21) - 1
        for view, (depth_cpu, (fx, fy, cx, cy), transform_cpu, excluded) in enumerate(views):
            height, width = map(int, depth_cpu.shape)
            depth = self._upload(("depth", view), depth_cpu).reshape(-1)
            valid = torch.isfinite(depth) & (depth >= self._min_depth_m) & (depth <= self._max_depth_m)
            if excluded is not None:
                valid &= ~excluded.reshape(-1).to(self.device)
            z = depth.clone()
            z.masked_fill_(~valid, 0.0)
            ray_x, ray_y = self._rays(height, width, fx, fy, cx, cy)
            points_camera = torch.empty((height * width, 3), dtype=torch.float32, device=self.device)
            points_camera[:, 0] = ray_x * z
            points_camera[:, 1] = ray_y * z
            points_camera[:, 2] = z
            transform = self._upload(("transform", view), transform_cpu)
            points_world = points_camera @ transform[:3, :3].T
            points_world.add_(transform[:3, 3])
            voxel_coords = torch.floor((points_world - self._origin) * self._inv_voxel_size).to(torch.int64)
            shifted = voxel_coords + bias
            in_key_range = (shifted >= 0).all(dim=1) & (shifted <= key_mask).all(dim=1)
            valid.logical_and_(in_key_range)
            keys = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
            keys.masked_fill_(~valid, rest_scene_kernel.INVALID_KEY)
            all_points.append(points_world)
            all_keys.append(keys)
        return self._dedup(torch.cat(all_points, dim=0), torch.cat(all_keys, dim=0))
