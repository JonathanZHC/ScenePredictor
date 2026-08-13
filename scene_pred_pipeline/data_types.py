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
class FlowInput:
    previous_candidates: torch.Tensor
    current_candidates: torch.Tensor
    previous_candidate_track_ids: torch.Tensor
    current_candidate_track_ids: torch.Tensor
    previous_anchors: torch.Tensor
    current_anchors: torch.Tensor
    previous_anchor_track_ids: torch.Tensor
    current_anchor_track_ids: torch.Tensor
    current_dense_points: torch.Tensor
    current_dense_track_ids: torch.Tensor
    common_track_ids: tuple[int, ...]
    dt_s: float


@dataclass
class FlowResult:
    source_anchors: torch.Tensor
    warped_anchors: torch.Tensor
    anchor_velocity: torch.Tensor


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
    tracked_masks: dict[str, np.ndarray]
    annotated_rgb: dict[str, np.ndarray]
    common_track_ids: tuple[int, ...]
    flow_valid: bool
    timings_ms: dict[str, float]
