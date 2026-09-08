from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch


@dataclass
class CameraFrameCpu:
    camera_name: str
    stamp_ns: int
    rgb: np.ndarray
    depth: np.ndarray
    K: np.ndarray
    T_world_camera: np.ndarray
    optical_frame_id: str


@dataclass
class MultiCameraFrame:
    stamp_ns: int
    cameras: dict[str, CameraFrameCpu]


@dataclass
class TrackedInstance:
    global_track_id: int
    semantic_label: str
    points_world: torch.Tensor


@dataclass
class TrackedInstanceFrame:
    frame_index: int
    stamp_ns: int
    instances: list[TrackedInstance]
    # All tracker output points live in one contiguous CUDA allocation.  The
    # per-instance tensors above are views into this storage, so downstream
    # stages can consume the full frame without rebuilding it with torch.cat().
    packed_points_world: torch.Tensor
    packed_track_ids: torch.Tensor
    # IDs are sorted during packing, making the common all-track case a true
    # zero-copy handoff to DifFlow.  Offsets are half-open [start, end) ranges.
    track_ids: tuple[int, ...]
    track_offsets: dict[int, tuple[int, int]]
    view_results: dict[str, Any] = field(default_factory=dict)
    tracker_timings_ms: dict[str, float] = field(default_factory=dict)
    tracker_metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class InstancePair:
    previous_stamp_ns: int
    current_stamp_ns: int
    common_track_ids: tuple[int, ...]
    previous_points: torch.Tensor
    current_points: torch.Tensor
    previous_track_ids: torch.Tensor
    current_track_ids: torch.Tensor
    dt_s: float


@dataclass
class FlowResult:
    # Only data consumed by recovery or ScenePredictor diagnostics is retained.
    # target_anchors, target_anchor_track_ids and anchor_velocity were redundant.
    source_anchors: torch.Tensor
    warped_anchors: torch.Tensor
    anchor_flow: torch.Tensor
    source_anchor_track_ids: torch.Tensor
    # Mask over InstancePair.current_points. Rejected dense points must not be
    # passed to motion recovery, otherwise they still receive velocities.
    target_input_keep_mask: torch.Tensor


@dataclass
class SceneVelocityOutput:
    stamp_ns: int
    # Timestamp delta used to convert the current flow pair into velocity.
    # Visualization reuses it so arrow length represents a configurable number
    # of frame intervals instead of a fixed arbitrary time scale.
    flow_dt_s: float
    tracked_points: torch.Tensor
    tracked_track_ids: torch.Tensor
    flow_points: torch.Tensor
    flow_velocity: torch.Tensor
    flow_track_ids: torch.Tensor
    source_anchors: torch.Tensor
    warped_anchors: torch.Tensor
    removed_outlier_points: torch.Tensor
    # Keep lightweight tracker result references instead of eagerly building
    # masks/overlays.  RosVisualizer only materializes them when a subscriber is
    # actually connected.
    view_results: dict[str, Any]
    common_track_ids: tuple[int, ...]
    flow_valid: bool
    timings_ms: dict[str, float]
    # Resolved after the existing end-of-cycle CUDA fence. The nested
    # statistics mapping is empty when adaptive voxel-2 was bypassed.
    outlier_filter_info: dict[str, Any] = field(default_factory=dict)
    # Populated only when outlier_filter.detailed_output is enabled. It is text
    # diagnostics only; removed-point visualization is independent of it.
    outlier_debug: dict[str, Any] = field(default_factory=dict)
    # Merged full-scene cloud (see SceneCloudConfig / scene_assembly.py). CUDA
    # tensors in the world frame; this is the in-process interface consumed by
    # the safety filter. None when scene_cloud.enabled is false.
    scene_points: torch.Tensor | None = None        # [N, 3] float32, meters
    scene_velocity: torch.Tensor | None = None      # [N, 3] float32, m/s (0 for rest points)
    scene_track_ids: torch.Tensor | None = None     # [N] int32 (0 for rest points)
    scene_num_dynamic: int = 0                      # rows [0, n) carry recovered velocity
