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


@dataclass
class SceneVelocityOutput:
    stamp_ns: int
    tracked_points: torch.Tensor
    tracked_track_ids: torch.Tensor
    flow_points: torch.Tensor
    flow_velocity: torch.Tensor
    flow_track_ids: torch.Tensor
    source_anchors: torch.Tensor
    warped_anchors: torch.Tensor
    # Keep lightweight tracker result references instead of eagerly building
    # masks/overlays.  RosVisualizer only materializes them when a subscriber is
    # actually connected.
    view_results: dict[str, Any]
    common_track_ids: tuple[int, ...]
    flow_valid: bool
    timings_ms: dict[str, float]
