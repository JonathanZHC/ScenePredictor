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
        self._origin: torch.Tensor | None = None
        self._inv_voxel_size = 0.0
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
        if output.flow_valid and output.flow_points.shape[0] > 0:
            dyn_points = output.flow_points
            dyn_velocity = output.flow_velocity
            dyn_ids = output.flow_track_ids
        else:
            dyn_points = output.tracked_points
            dyn_velocity = torch.zeros_like(dyn_points)
            dyn_ids = output.tracked_track_ids
        dyn_points = dyn_points.to(self.device, torch.float32)
        dyn_velocity = dyn_velocity.to(self.device, torch.float32)
        dyn_ids = dyn_ids.to(self.device, torch.int32)

        if self.scene_cfg.include_rest_points:
            if output.view_results:
                rest = self.rest_scene_points(output)
            elif frame is not None:
                rest = self.rest_points_from_frame(frame)
            else:
                rest = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        else:
            rest = torch.empty((0, 3), dtype=torch.float32, device=self.device)

        points = torch.cat((dyn_points, rest), dim=0)
        velocity = torch.cat((dyn_velocity, torch.zeros_like(rest)), dim=0)
        track_ids = torch.cat((dyn_ids, torch.zeros(rest.shape[0], dtype=torch.int32, device=self.device)), dim=0)
        num_dynamic = int(dyn_points.shape[0])

        if self._workspace is not None and points.shape[0] > 0:
            lo, hi = self._workspace
            keep = ((points >= lo) & (points <= hi)).all(dim=1)
            # Keep the dynamic-first ordering; count dynamic survivors.
            num_dynamic = int(keep[:num_dynamic].sum().item())
            index = keep.nonzero(as_tuple=True)[0]
            points = points.index_select(0, index)
            velocity = velocity.index_select(0, index)
            track_ids = track_ids.index_select(0, index)

        output.scene_points = points.contiguous()
        output.scene_velocity = velocity.contiguous()
        output.scene_track_ids = track_ids.contiguous()
        output.scene_num_dynamic = num_dynamic

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
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m

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

    def _exclusion_mask_gpu(self, result: Any, shape: tuple[int, int]) -> torch.Tensor:
        """Union of tracked final masks + the tracker-provided exclusion-only mask."""
        excluded_cpu = np.zeros(shape, dtype=bool)
        gpu_masks: list[torch.Tensor] = []
        for instance in getattr(result, "instances", ()):
            mask = getattr(instance, "mask", None)
            if mask is not None and not torch.is_tensor(mask):
                value = np.asarray(mask)
                if value.shape == shape:
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
                np.logical_or(excluded_cpu, value, out=excluded_cpu)

        excluded = torch.as_tensor(excluded_cpu, dtype=torch.bool, device=self.device)
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
            excluded = self._exclusion_mask_gpu(result, tuple(depth_cpu.shape))
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

    def _rest_points_from_views(self, views) -> torch.Tensor:
        """Dense backprojection of every view, world transform, voxel dedup on one lattice; all on CUDA.

        views: iterable of (depth_m [H,W] float32 numpy, (fx, fy, cx, cy), world_from_camera 4x4/3x4, excluded mask [H,W] bool tensor or None)
        """
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if self._origin is None:
            return empty
        all_points: list[torch.Tensor] = []
        all_keys: list[torch.Tensor] = []
        bias = 1 << 20
        key_mask = (1 << 21) - 1
        invalid_key = torch.iinfo(torch.int64).max
        for depth_cpu, (fx, fy, cx, cy), transform_cpu, excluded in views:
            if transform_cpu.shape not in {(4, 4), (3, 4)}:
                continue
            height, width = map(int, depth_cpu.shape)
            depth = torch.as_tensor(depth_cpu, dtype=torch.float32, device=self.device).reshape(-1)
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
            transform = torch.as_tensor(transform_cpu, dtype=torch.float32, device=self.device)
            points_world = points_camera @ transform[:3, :3].T
            points_world.add_(transform[:3, 3])
            voxel_coords = torch.floor((points_world - self._origin) * self._inv_voxel_size).to(torch.int64)
            shifted = voxel_coords + bias
            in_key_range = (shifted >= 0).all(dim=1) & (shifted <= key_mask).all(dim=1)
            valid.logical_and_(in_key_range)
            keys = (shifted[:, 0] << 42) | (shifted[:, 1] << 21) | shifted[:, 2]
            keys.masked_fill_(~valid, invalid_key)
            all_points.append(points_world)
            all_keys.append(keys)
        if not all_points:
            return empty
        points = all_points[0] if len(all_points) == 1 else torch.cat(all_points, dim=0)
        keys = all_keys[0] if len(all_keys) == 1 else torch.cat(all_keys, dim=0)
        sorted_keys, order = torch.sort(keys)
        keep = torch.empty_like(sorted_keys, dtype=torch.bool)
        keep[0] = sorted_keys[0] != invalid_key
        if sorted_keys.numel() > 1:
            keep[1:] = (sorted_keys[1:] != sorted_keys[:-1]) & (sorted_keys[1:] != invalid_key)
        return points[order[keep]]
