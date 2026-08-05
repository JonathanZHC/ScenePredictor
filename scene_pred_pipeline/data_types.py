from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum

import numpy as np
import torch


class MotionState(IntEnum):
    STATIC = 0
    MOVING = 1


class PointType(IntEnum):
    BACKGROUND = 0
    STATIC_OBJECT = 1
    MOVING_OBJECT = 2


@dataclass(frozen=True)
class ImageDetection:
    bbox_xyxy: tuple[int, int, int, int]
    class_id: int
    class_name: str
    confidence: float
    track_id: int
    motion_state: MotionState


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
class CameraFrameGpu:
    camera_name: str
    stamp_ns: int
    depth: torch.Tensor
    K: torch.Tensor
    T_world_camera: torch.Tensor
    T_camera_world: torch.Tensor
    optical_frame_id: str


@dataclass
class MultiCameraFrame:
    stamp_ns: int
    cameras: dict[str, CameraFrameCpu]


@dataclass
class ViewInstance:
    camera_name: str
    local_instance_id: int

    class_id: int
    class_name: str
    class_confidence: float
    bbox_xyxy: tuple[int, int, int, int]

    mask_original: torch.Tensor
    mask_eroded: torch.Tensor

    pcd_world: torch.Tensor
    centroid_world: torch.Tensor
    aabb_min_world: torch.Tensor
    aabb_max_world: torch.Tensor
    reprojection_points_world: torch.Tensor


@dataclass
class PerViewResult:
    camera: CameraFrameGpu
    instances: list[ViewInstance]
    background_pcd_world: torch.Tensor


@dataclass
class FusedObject:
    frame_object_id: int
    class_id: int
    class_name: str
    class_confidence: float

    pcd_world: torch.Tensor
    centroid_world: torch.Tensor
    aabb_min_world: torch.Tensor
    aabb_max_world: torch.Tensor

    source_camera_names: tuple[str, ...]
    members: list[ViewInstance]


@dataclass
class FusedFrame:
    stamp_ns: int
    objects: list[FusedObject]
    background_pcd_world: torch.Tensor
    camera_results: dict[str, PerViewResult]


@dataclass
class VoxelizedObject:
    frame_object_id: int
    class_id: int
    class_name: str
    class_confidence: float

    points: torch.Tensor
    voxel_keys: torch.Tensor
    voxel_coords: torch.Tensor

    centroid_world: torch.Tensor
    aabb_min_world: torch.Tensor
    aabb_max_world: torch.Tensor
    members: list[ViewInstance]

    track_id: int = -1
    motion_state: MotionState = MotionState.STATIC


@dataclass
class VoxelizedFrame:
    frame_index: int
    stamp_ns: int
    objects: list[VoxelizedObject]
    background_points: torch.Tensor
    background_keys: torch.Tensor
    camera_results: dict[str, PerViewResult]


@dataclass
class TrackObservation:
    frame_index: int
    stamp_ns: int
    centroid_world: torch.Tensor
    voxel_keys: torch.Tensor
    points: torch.Tensor


@dataclass
class Track:
    track_id: int
    last_object: VoxelizedObject
    history: dict[int, TrackObservation] = field(default_factory=dict)


@dataclass
class FlowCache:
    """Per-frame moving-point sampling cache.

    The current frame is preselected and FPS-sampled exactly once. On the next
    cycle these tensors are reused directly as the previous-frame input.
    """

    candidates: torch.Tensor
    candidate_track_ids: torch.Tensor
    anchors: torch.Tensor
    anchor_track_ids: torch.Tensor


@dataclass
class TrackedFrame:
    frame_index: int
    stamp_ns: int
    objects: list[VoxelizedObject]
    background_points: torch.Tensor
    background_keys: torch.Tensor
    moving_masks: dict[str, torch.Tensor]
    flow_cache: FlowCache | None = None


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

    dt_s: float


@dataclass
class FlowResult:
    source_anchors: torch.Tensor
    warped_anchors: torch.Tensor
    anchor_velocity: torch.Tensor


@dataclass
class SceneVelocityOutput:
    stamp_ns: int

    background_points: torch.Tensor
    static_points: torch.Tensor

    moving_points: torch.Tensor
    moving_velocity: torch.Tensor
    moving_track_ids: torch.Tensor

    moving_masks: dict[str, torch.Tensor]
    camera_rgb: dict[str, np.ndarray]
    annotated_rgb: dict[str, np.ndarray]

    timings_ms: dict[str, float]
