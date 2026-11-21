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
    # Optional: the depth image as a CUDA tensor [H, W] float32 (meters), set by a producer that already
    # has it on the device (an upstream depth filter). Consumers that can take a tensor use it and skip
    # the upload; `depth` is then a LazyHostDepth on the same tensor, downloaded only if somebody reads it.
    depth_gpu: "object | None" = None
    # Optional bool [H, W] (host array or CUDA tensor): pixels whose depth was removed because the ROBOT is there (the
    # upstream self-filter). The static memory keeps remembered points behind these pixels without a timeout; pixels
    # without depth for any other reason (holes, out of range) are "unknown" and expire.
    occluder_mask: "object | None" = None
    # Set by the pipeline's zone (zone.py): the depth image as it arrived, before the points outside the zone were
    # blanked in depth / depth_gpu. Read by the static memory's visibility test, which must see past the zone.
    depth_unzoned_gpu: "object | None" = None


class LazyHostDepth:
    """Host view of a depth image that lives on the GPU: looks like the numpy array, downloads on first use.

    Shape, dtype, ndim and size are known without a download. Reading the pixels (np.asarray, indexing, any
    other array attribute) downloads the whole image once; later reads reuse it. Thread-safe. Arithmetic on
    the object itself is not defined: convert with np.asarray first.
    """

    lazy_host = True

    def __init__(self, tensor) -> None:
        import threading

        self._tensor = tensor
        self._array = None
        self._lock = threading.Lock()
        self.shape = tuple(int(n) for n in tensor.shape)
        self.ndim = len(self.shape)
        self.size = int(np.prod(self.shape)) if self.shape else 1
        self.dtype = np.dtype(np.float32)

    @property
    def materialized(self) -> bool:
        return self._array is not None

    def host(self) -> np.ndarray:
        if self._array is None:
            with self._lock:
                if self._array is None:
                    self._array = np.ascontiguousarray(self._tensor.detach().float().cpu().numpy(), dtype=np.float32)
        return self._array

    def __array__(self, dtype=None, copy=None):
        array = self.host()
        if dtype is not None and np.dtype(dtype) != array.dtype:
            return array.astype(dtype)
        return array.copy() if copy else array

    def __getitem__(self, key):
        return self.host()[key]

    def __len__(self) -> int:
        return self.shape[0]

    def __getattr__(self, name):                 # anything else an ndarray offers
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self.host(), name)


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
    # Only with the DifFlow uncertainty head, rows aligned with anchor_flow:
    # covariance Sigma of each anchor's one-pair displacement [M, 3, 3] (m^2) and
    # the scale s [M] of the configured calibration region at the configured level
    # (ellipsoid: d^T Sigma^-1 d <= s^2; ball: |d| <= s det(Sigma)^(1/6); Mondrian
    # bin of |anchor_flow| or the global value; see DifFlowUncertaintyCalibrationConfig).
    anchor_flow_cov: torch.Tensor | None = None
    anchor_flow_cov_scale: torch.Tensor | None = None


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
    # Points that entered the scene assembly this cycle: every dynamic point plus every valid,
    # non-excluded depth pixel of every camera BEFORE the voxel dedup / workspace crop, i.e. the
    # raw perception count that scene_points was downsampled from. A 0-d CUDA tensor when the
    # rest scene was built (kept on the device to avoid a host sync; int() it after the cycle fence).
    scene_num_input: int = 0
    # Host copy of the global track ids of the tracked instances this cycle (lets the scene
    # assembly skip a device check when every tracked instance has a flow estimate).
    tracked_instance_ids: tuple[int, ...] = ()
    # Track id of every flow anchor (rows of source_anchors / warped_anchors).
    anchor_track_ids: torch.Tensor | None = None
    # Flow uncertainty, rows aligned with warped_anchors; None without the DifFlow
    # uncertainty head or without a valid flow pair (see FlowResult).
    anchor_flow_cov: torch.Tensor | None = None          # [M, 3, 3] float32, m^2
    anchor_flow_cov_scale: torch.Tensor | None = None    # [M] calibration scale s (FlowResult)
    # Calibrated velocity region of every flow point (rows of flow_points; only when
    # ScenePredictionPipeline.compute_velocity_region is set): the true velocity lies in
    # {flow_velocity + d : d^T R^-1 d <= 1} (velocity_recovery.anchor_velocity_region).
    flow_velocity_region: torch.Tensor | None = None     # [Nf, 3, 3] float32, m^2/s^2
    # The same for the first scene_velocity_region.shape[0] rows of scene_points (the moving
    # points; the rest scene has no region).
    scene_velocity_region: torch.Tensor | None = None    # [Nd, 3, 3] float32, m^2/s^2
