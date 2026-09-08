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
excluded masks, voxel-deduplicated on the tracker's shared world lattice, with
zero velocity and track id 0. This is the code that used to live in
RosVisualizer as a visualization-only path.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .config import PipelineConfig
from .data_types import SceneVelocityOutput


class SceneCloudAssembler:
    def __init__(self, config: PipelineConfig, *, tracker_config: Any) -> None:
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
    def assemble(self, output: SceneVelocityOutput) -> None:
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
            rest = self.rest_scene_points(output)
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
    def _configure_rest_scene(self, tracker_config: Any) -> None:
        try:
            voxel_size_m = float(tracker_config.shared_voxel_grid.voxel_size_m)
            origin_world = np.asarray(tracker_config.shared_voxel_grid.origin_world, dtype=np.float32).reshape(3)
            min_depth_m = float(tracker_config.postprocess.min_valid_depth_m)
            max_depth_m = float(tracker_config.postprocess.max_valid_depth_m)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Rest-scene assembly requires tracker shared_voxel_grid.voxel_size_m/"
                "origin_world and postprocess min/max depth settings."
            ) from exc
        if voxel_size_m <= 0.0:
            raise ValueError(f"shared_voxel_grid.voxel_size_m must be positive, got {voxel_size_m}")
        if max_depth_m < min_depth_m:
            raise ValueError("postprocess.max_valid_depth_m must be >= min_valid_depth_m")
        self._inv_voxel_size = 1.0 / voxel_size_m
        self._origin = torch.as_tensor(origin_world, dtype=torch.float32, device=self.device).contiguous()
        self._min_depth_m = min_depth_m
        self._max_depth_m = max_depth_m

    def _rays(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        intrinsics = frame.intrinsics
        height, width = map(int, frame.depth_m.shape)
        key = (height, width, float(intrinsics.fx), float(intrinsics.fy), float(intrinsics.cx), float(intrinsics.cy), str(self.device))
        cached = self._ray_cache.get(key)
        if cached is not None:
            return cached
        x_ray = (torch.arange(width, dtype=torch.float32, device=self.device) - float(intrinsics.cx)) / float(intrinsics.fx)
        y_ray = (torch.arange(height, dtype=torch.float32, device=self.device) - float(intrinsics.cy)) / float(intrinsics.fy)
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
        """Rest-scene cloud on the tracker's world lattice, entirely on CUDA."""
        empty = torch.empty((0, 3), dtype=torch.float32, device=self.device)
        if self._origin is None:
            return empty
        all_points: list[torch.Tensor] = []
        all_keys: list[torch.Tensor] = []
        bias = 1 << 20
        key_mask = (1 << 21) - 1
        invalid_key = torch.iinfo(torch.int64).max
        for result in output.view_results.values():
            frame = result.frame
            depth_cpu = np.asarray(frame.depth_m, dtype=np.float32)
            if depth_cpu.ndim != 2 or depth_cpu.size == 0:
                continue
            height, width = map(int, depth_cpu.shape)
            shape = (height, width)
            world_from_camera = getattr(frame, "world_from_camera", None)
            if world_from_camera is None:
                continue
            transform_cpu = np.asarray(world_from_camera, dtype=np.float32)
            if transform_cpu.shape not in {(4, 4), (3, 4)}:
                continue
            depth = torch.as_tensor(depth_cpu, dtype=torch.float32, device=self.device).reshape(-1)
            excluded = self._exclusion_mask_gpu(result, shape).reshape(-1)
            valid = (~excluded) & torch.isfinite(depth) & (depth >= self._min_depth_m) & (depth <= self._max_depth_m)
            z = depth.clone()
            z.masked_fill_(~valid, 0.0)
            ray_x, ray_y = self._rays(frame)
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
