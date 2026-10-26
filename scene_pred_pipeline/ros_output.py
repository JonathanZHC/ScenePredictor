from __future__ import annotations

import struct
import time
from typing import Any

import cv2
import numpy as np
import torch
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from sam_rgbd_tracking.visualization import instance_mask_cpu, make_overlay

from .config import PipelineConfig
from .data_types import SceneVelocityOutput


_EXCLUDED_OVERLAY_COLOR = np.asarray([255, 70, 220], dtype=np.uint8)


# Tracked-instance colors. Red is deliberately absent: downstream (safety filter)
# reserves red for the critical CBF constraint points.
_PALETTE = np.asarray(
    [
        [245, 150, 40],
        [80, 210, 120],
        [80, 150, 240],
        [235, 220, 70],
        [180, 90, 225],
        [70, 210, 215],
    ],
    dtype=np.uint8,
)


# Flow-uncertainty regions: viridis from 0.3 to 1.0 (perceptually ordered, no
# red; the dark violet start is dropped so translucent markers stay visible on
# RViz's dark background), spanning _UNCERTAINTY_COLOR_RADIUS_M on a log scale.
_UNCERTAINTY_RAMP = np.asarray(
    [
        [0.207, 0.372, 0.553],
        [0.164, 0.471, 0.558],
        [0.128, 0.567, 0.551],
        [0.135, 0.659, 0.518],
        [0.267, 0.749, 0.441],
        [0.478, 0.821, 0.318],
        [0.741, 0.873, 0.150],
        [0.993, 0.906, 0.144],
    ],
    dtype=np.float64,
)
# Ramp ends of the one-frame equal-volume radius s det(Sigma)^(1/6) of the
# calibrated region (the ball radius; the geometric-mean semi-axis of the
# ellipsoid). On the real2 rep split (deployed path, ellipsoid, Mondrian, 0.95) it
# has p50 2.9 mm and p99 13.8 mm: a third of the anchors clip to the low end,
# fewer than 0.1 % reach the high end.
_UNCERTAINTY_COLOR_RADIUS_M = (0.002, 0.03)
# RViz warns about zero-size markers (velocity_marker_scale_factor 0, flat covariances).
_MIN_MARKER_DIAMETER_M = 1.0e-4


def _rotation_to_quaternion_xyzw(rotations: np.ndarray) -> np.ndarray:
    """Unit quaternions (x, y, z, w) of proper rotation matrices [K, 3, 3] (Shepperd)."""
    r = rotations
    trace = r[:, 0, 0] + r[:, 1, 1] + r[:, 2, 2]
    # Divide by the largest of |w|, |x|, |y|, |z| (4 w^2 = 1 + trace,
    # 4 x^2 = 1 + 2 r00 - trace, ...): pick the largest of trace, r00, r11, r22.
    choice = np.argmax(np.stack((trace, r[:, 0, 0], r[:, 1, 1], r[:, 2, 2]), axis=1), axis=1)
    q = np.empty((r.shape[0], 4), dtype=np.float64)

    m = choice == 0
    s = 2.0 * np.sqrt(np.maximum(1.0 + trace[m], 1.0e-12))
    q[m, 0] = (r[m, 2, 1] - r[m, 1, 2]) / s
    q[m, 1] = (r[m, 0, 2] - r[m, 2, 0]) / s
    q[m, 2] = (r[m, 1, 0] - r[m, 0, 1]) / s
    q[m, 3] = 0.25 * s

    m = choice == 1
    s = 2.0 * np.sqrt(np.maximum(1.0 + r[m, 0, 0] - r[m, 1, 1] - r[m, 2, 2], 1.0e-12))
    q[m, 0] = 0.25 * s
    q[m, 1] = (r[m, 0, 1] + r[m, 1, 0]) / s
    q[m, 2] = (r[m, 0, 2] + r[m, 2, 0]) / s
    q[m, 3] = (r[m, 2, 1] - r[m, 1, 2]) / s

    m = choice == 2
    s = 2.0 * np.sqrt(np.maximum(1.0 + r[m, 1, 1] - r[m, 0, 0] - r[m, 2, 2], 1.0e-12))
    q[m, 0] = (r[m, 0, 1] + r[m, 1, 0]) / s
    q[m, 1] = 0.25 * s
    q[m, 2] = (r[m, 1, 2] + r[m, 2, 1]) / s
    q[m, 3] = (r[m, 0, 2] - r[m, 2, 0]) / s

    m = choice == 3
    s = 2.0 * np.sqrt(np.maximum(1.0 + r[m, 2, 2] - r[m, 0, 0] - r[m, 1, 1], 1.0e-12))
    q[m, 0] = (r[m, 0, 2] + r[m, 2, 0]) / s
    q[m, 1] = (r[m, 1, 2] + r[m, 2, 1]) / s
    q[m, 2] = 0.25 * s
    q[m, 3] = (r[m, 1, 0] - r[m, 0, 1]) / s

    return q / np.linalg.norm(q, axis=1, keepdims=True)


def ellipsoid_marker_geometry(
    centers: np.ndarray,
    covs: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pose and size of SPHERE markers drawing the 1-sigma ellipsoids of ``covs``.

    centers [K, 3] and finite symmetric PSD covs [K, 3, 3] (any consistent unit).
    Returns float64 positions [K, 3], quaternions_xyzw [K, 4] (the eigenvector
    frame as a proper, right-handed rotation R) and diameters [K, 3] (full axis
    lengths 2 sqrt(eigenvalue) along the marker's local x/y/z), such that
    R diag((diameters / 2)^2) R^T == cov.
    """
    positions = np.array(centers, dtype=np.float64).reshape(-1, 3)
    covs = np.asarray(covs, dtype=np.float64).reshape(-1, 3, 3)
    eigenvalues, frames = np.linalg.eigh(0.5 * (covs + np.swapaxes(covs, 1, 2)))
    # eigh may return a reflection; flipping one eigenvector keeps the ellipsoid.
    frames[np.linalg.det(frames) < 0.0, :, 2] *= -1.0
    diameters = 2.0 * np.sqrt(np.clip(eigenvalues, 0.0, None))
    return positions, _rotation_to_quaternion_xyzw(frames), diameters


def uncertainty_region_markers(
    region: str,
    centers: np.ndarray,
    covs: np.ndarray,
    scales: np.ndarray,
    factor: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """SPHERE marker poses of calibrated one-frame regions, scaled like the velocity lines.

    covs [K, 3, 3] (m^2) and scales [K] are Sigma and the calibration scale s of
    ``region`` of the (nearest) anchors, centers [K, 3] the marker centres:

    * ``ellipsoid`` (d^T Sigma^-1 d <= s^2): the ellipsoid of covariance
      factor^2 s^2 Sigma;
    * ``ball`` (|d| <= s det(Sigma)^(1/6)): a sphere of radius
      factor s det(Sigma)^(1/6).

    The size is calibrated for the anchor's own displacement; at a velocity-line
    tip it is a constant-velocity forecast (not calibrated).
    Returns positions [K, 3], quaternions_xyzw [K, 4] and diameters [K, 3] as
    ellipsoid_marker_geometry, plus the one-frame equal-volume radius
    s det(Sigma)^(1/6) [K] (m) of the region, independent of factor (marker colour).
    """
    covs = np.asarray(covs, dtype=np.float64).reshape(-1, 3, 3)
    scales = np.asarray(scales, dtype=np.float64).reshape(-1)
    radius = scales * np.clip(np.linalg.det(covs), 0.0, None) ** (1.0 / 6.0)
    if region == "ellipsoid":
        positions, quaternions, diameters = ellipsoid_marker_geometry(
            centers, (factor * scales)[:, None, None] ** 2 * covs
        )
    elif region == "ball":
        positions = np.array(centers, dtype=np.float64).reshape(-1, 3)
        quaternions = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), (positions.shape[0], 1))
        diameters = np.repeat((2.0 * factor * radius)[:, None], 3, axis=1)
    else:
        raise ValueError(f"Unknown calibration region {region!r} (ellipsoid or ball)")
    return positions, quaternions, diameters, radius


def uncertainty_colors(radius_m: np.ndarray) -> np.ndarray:
    """RGB [K, 3] in [0, 1] for equal-volume radii on the non-red uncertainty ramp."""
    low, high = _UNCERTAINTY_COLOR_RADIUS_M
    radius = np.clip(np.nan_to_num(np.asarray(radius_m, dtype=np.float64), nan=high), low, high)
    position = np.log(radius / low) / np.log(high / low) * (len(_UNCERTAINTY_RAMP) - 1)
    index = np.minimum(position.astype(np.int64), len(_UNCERTAINTY_RAMP) - 2)
    weight = (position - index)[:, None]
    return (1.0 - weight) * _UNCERTAINTY_RAMP[index] + weight * _UNCERTAINTY_RAMP[index + 1]


def _rate_gate(next_ns: int, period_ns: int, now_ns: int) -> tuple[bool, int]:
    """Phase-locked rate gate: (due, next deadline). Due at the first call of every period, so
    frames faster than the rate give the configured average rate; scheduling ``now + period``
    instead loses up to one frame interval per publish (10 Hz -> 9 Hz with 27 Hz frames).
    After a pause longer than one period it restarts from now (no burst)."""
    if now_ns < next_ns:
        return False, next_ns
    next_ns += period_ns
    if next_ns <= now_ns:
        next_ns = now_ns + period_ns
    return True, next_ns


def _evenly_spread(count: int, limit: int) -> np.ndarray:
    """At most ``limit`` indices of range(count), evenly spread and including both ends."""
    if count <= limit:
        return np.arange(count, dtype=np.int64)
    return np.unique(np.linspace(0, count - 1, max(1, limit)).round().astype(np.int64))


def _stamp_message(node: Node, stamp_ns: int):
    message = node.get_clock().now().to_msg()
    message.sec = int(stamp_ns) // 1_000_000_000
    message.nanosec = int(stamp_ns) % 1_000_000_000
    return message


def _rgb_float(red: int, green: int, blue: int) -> float:
    packed = (int(red) << 16) | (int(green) << 8) | int(blue)
    return struct.unpack("f", struct.pack("I", packed))[0]


def _xyzrgb_cloud(
    node: Node,
    points: torch.Tensor,
    stamp_ns: int,
    frame_id: str,
    rgb: tuple[int, int, int],
) -> PointCloud2:
    cpu = points.detach().float().cpu().numpy()
    packed = np.full((cpu.shape[0], 1), _rgb_float(*rgb), dtype=np.float32)
    cloud = np.ascontiguousarray(np.concatenate((cpu, packed), axis=1))
    message = PointCloud2()
    message.header = Header(stamp=_stamp_message(node, stamp_ns), frame_id=frame_id)
    message.height = 1
    message.width = cloud.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = 16 * cloud.shape[0]
    message.data = cloud.tobytes()
    message.is_dense = True
    return message


def _xyz_cloud(
    node: Node,
    points: torch.Tensor,
    stamp_ns: int,
    frame_id: str,
) -> PointCloud2:
    """Pack an XYZ-only cloud; RViz supplies the fixed display color.

    Rest-scene visualization intentionally omits a per-point RGB field.  This
    reduces PointCloud2 payload from 16 to 12 bytes/point while preserving the
    requested all-white appearance through RViz's FlatColor transformer.
    """
    cpu = np.ascontiguousarray(points.detach().float().cpu().numpy(), dtype=np.float32)
    message = PointCloud2()
    message.header = Header(stamp=_stamp_message(node, stamp_ns), frame_id=frame_id)
    message.height = 1
    message.width = cpu.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = 12 * cpu.shape[0]
    message.data = cpu.tobytes()
    message.is_dense = True
    return message


def _tracked_cloud(
    node: Node,
    output: SceneVelocityOutput,
    frame_id: str,
) -> PointCloud2:
    points = output.tracked_points.detach().float().cpu().numpy()
    track_ids = output.tracked_track_ids.detach().to(torch.int32).cpu().numpy()
    colors = np.zeros((len(track_ids),), dtype=np.float32)
    for track_id in np.unique(track_ids):
        color = _PALETTE[(int(track_id) - 1) % len(_PALETTE)]
        colors[track_ids == track_id] = _rgb_float(*color.tolist())

    structured = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("rgb", "<f4"), ("track_id", "<i4"),
        ],
    )
    if points.shape[0]:
        structured["x"], structured["y"], structured["z"] = points.T
    structured["rgb"] = colors
    structured["track_id"] = track_ids

    message = PointCloud2()
    message.header = Header(stamp=_stamp_message(node, output.stamp_ns), frame_id=frame_id)
    message.height = 1
    message.width = points.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="rgb", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="track_id", offset=16, datatype=PointField.INT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = structured.dtype.itemsize
    message.row_step = message.point_step * points.shape[0]
    message.data = structured.tobytes()
    message.is_dense = True
    return message


def _velocity_cloud(
    node: Node,
    points_gpu: torch.Tensor,
    velocity_gpu: torch.Tensor,
    track_ids_gpu: torch.Tensor,
    stamp_ns: int,
    frame_id: str,
) -> PointCloud2:
    """x y z vx vy vz track_id, 28 bytes per point (the on-the-wire scene-cloud layout)."""
    points = points_gpu.detach().float().cpu().numpy()
    velocity = velocity_gpu.detach().float().cpu().numpy()
    track_ids = track_ids_gpu.detach().to(torch.int32).cpu().numpy()
    structured = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"),
            ("track_id", "<i4"),
        ],
    )
    if points.shape[0]:
        structured["x"], structured["y"], structured["z"] = points.T
        structured["vx"], structured["vy"], structured["vz"] = velocity.T
    structured["track_id"] = track_ids

    message = PointCloud2()
    message.header = Header(stamp=_stamp_message(node, stamp_ns), frame_id=frame_id)
    message.height = 1
    message.width = points.shape[0]
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
        PointField(name="vx", offset=12, datatype=PointField.FLOAT32, count=1),
        PointField(name="vy", offset=16, datatype=PointField.FLOAT32, count=1),
        PointField(name="vz", offset=20, datatype=PointField.FLOAT32, count=1),
        PointField(name="track_id", offset=24, datatype=PointField.INT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = structured.dtype.itemsize
    message.row_step = message.point_step * points.shape[0]
    message.data = structured.tobytes()
    message.is_dense = True
    return message


def _rgb8_image(node: Node, image: np.ndarray, stamp_ns: int, frame_id: str) -> Image:
    image = np.ascontiguousarray(image, dtype=np.uint8)
    message = Image()
    message.header.stamp = _stamp_message(node, stamp_ns)
    message.header.frame_id = frame_id
    message.height = image.shape[0]
    message.width = image.shape[1]
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = image.shape[1] * 3
    message.data = image.tobytes()
    return message


class RosVisualizer:
    """Publish diagnostics only when they have an active ROS subscriber.

    This keeps all expensive visualization work -- D2H copies, overlay drawing,
    mask merging, PointCloud2 serialization and Marker construction -- outside
    the numerical critical path when RViz/other consumers are disconnected.
    """

    def __init__(
        self,
        node: Node,
        config: PipelineConfig,
        *,
        tracker_config: Any | None = None,
    ) -> None:
        self.node = node
        self.config = config
        # RELIABLE + keep-last(1): a reliable publisher matches subscribers that ask for either
        # reliability (a best-effort publisher is refused by reliable subscribers, e.g. an RViz
        # display added by hand, which defaults to reliable: "requesting incompatible QoS ...
        # No messages will be sent"). Depth 1 keeps it latest-only; on one host the transport is
        # shared memory, so reliability costs nothing measurable.
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        output = config.output

        # One cheap global gate controls every visualization topic so all enabled
        # RViz displays stay on the same SceneVelocityOutput/frame. The gate runs
        # before subscription checks and before any D2H/serialization/geometry.
        publish_hz = float(output.visualization_publish_hz)
        self._visualization_period_ns = (
            max(1, int(round(1_000_000_000.0 / publish_hz)))
            if publish_hz > 0.0
            else 0
        )
        self._next_visualization_ns = 0
        # The flow-uncertainty markers (one Python Marker each) get their own rate, checked
        # only on frames that passed the global gate. At or above the global rate they go out
        # with every global publish, i.e. on the same frames as the velocity lines.
        self._uncertainty_period_ns = max(
            1, int(round(1_000_000_000.0 / float(output.uncertainty_marker_publish_hz)))
        )
        self._uncertainty_with_every_publish = (
            0 < self._visualization_period_ns
            and self._uncertainty_period_ns <= self._visualization_period_ns
        )
        self._next_uncertainty_ns = 0
        self._uncertainty_region = str(config.difflow.uncertainty_head.calibration.region)
        # Reused Marker objects of the uncertainty markers (publisher thread only; rclpy
        # serialises on publish): ns/id/type/action/alpha are set once per pool entry.
        self._uncertainty_marker_pool: list[Marker] = []
        # Real publishes per topic since the last rate_summary() (profiler summary line).
        self._publish_counts: dict[str, int] = {}
        self._rate_since = time.monotonic()

        self.tracked_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/tracked_points", qos)
            if output.publish_tracked_objects else None
        )
        self.rest_scene_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/rest_points", qos)
            if output.publish_rest_scene else None
        )
        self.velocity_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/scene_velocity", qos)
            if output.publish_velocity_cloud else None
        )
        # Throttled ROS copy of the merged full-scene cloud (same 28-byte layout as
        # scene_velocity, track_id 0 = rest scene). The in-process consumer uses
        # SceneVelocityOutput.scene_points/scene_velocity directly.
        scene_cloud_hz = float(config.scene_cloud.publish_hz)
        self.scene_cloud_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/scene_cloud", qos)
            if config.scene_cloud.enabled and scene_cloud_hz > 0.0 else None
        )
        self._scene_cloud_period_ns = int(round(1_000_000_000.0 / scene_cloud_hz)) if scene_cloud_hz > 0.0 else 0
        self._next_scene_cloud_ns = 0
        self.source_anchor_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/flow/source_anchors", qos)
            if output.publish_flow_anchors else None
        )
        self.warped_anchor_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/flow/warped_anchors", qos)
            if output.publish_flow_anchors else None
        )
        self.outlier_removed_pub = (
            node.create_publisher(
                PointCloud2, "/scene_predictor/outlier_filter/removed_points", qos
            )
            if output.publish_removed_outlier_points else None
        )
        self.marker_pub = (
            node.create_publisher(MarkerArray, "/scene_predictor/velocity_markers", qos)
            if output.publish_velocity_markers else None
        )
        # On by default, but only with an uncertainty head (no covariance, no topic).
        self.uncertainty_marker_pub = (
            node.create_publisher(
                MarkerArray, "/scene_predictor/flow_uncertainty_markers", qos
            )
            if output.publish_uncertainty_markers and config.difflow.uncertainty_head.enable
            else None
        )
        self.annotated_rgb_pubs = (
            {
                camera: node.create_publisher(
                    Image,
                    f"/scene_predictor/{camera}/rgb_annotated",
                    qos,
                )
                for camera in config.ros.camera_names
            }
            if output.publish_annotated_rgb else {}
        )
        self.mask_pubs = (
            {
                camera: node.create_publisher(
                    Image,
                    f"/scene_predictor/{camera}/tracked_mask",
                    qos,
                )
                for camera in config.ros.camera_names
            }
            if output.publish_tracked_masks else {}
        )


    @staticmethod
    def _subscribed(publisher) -> bool:
        return publisher is not None and int(publisher.get_subscription_count()) > 0

    def _visualization_publish_due(self) -> bool:
        """Return True at the configured global visualization update rate.

        This is intentionally a monotonic integer-clock check in the pipeline
        worker thread: no ROS timer, lock, CUDA synchronization, or retained output
        is needed. A zero period preserves the original publish-every-frame behavior.
        """
        period_ns = self._visualization_period_ns
        if period_ns <= 0:
            return True

        due, self._next_visualization_ns = _rate_gate(
            self._next_visualization_ns, period_ns, time.monotonic_ns()
        )
        return due

    def _uncertainty_markers_due(self) -> bool:
        """uncertainty_marker_publish_hz gate, applied on top of the global gate."""
        if self.uncertainty_marker_pub is None:
            return False
        if self._uncertainty_with_every_publish:
            return True
        due, self._next_uncertainty_ns = _rate_gate(
            self._next_uncertainty_ns, self._uncertainty_period_ns, time.monotonic_ns()
        )
        return due

    def _count_publish(self, topic: str) -> None:
        self._publish_counts[topic] = self._publish_counts.get(topic, 0) + 1

    def rate_summary(self) -> str:
        """Real publish rates of the marker topics since the previous call (one summary line)."""
        now = time.monotonic()
        elapsed = max(now - self._rate_since, 1.0e-9)
        rates = ", ".join(
            f"{topic} {count / elapsed:.1f} Hz"
            for topic, count in sorted(self._publish_counts.items())
        )
        self._publish_counts = {}
        self._rate_since = now
        return f"visualization publish rate: {rates or 'none'}"

    def _publish_masks(self, output: SceneVelocityOutput) -> None:
        for camera, result in output.view_results.items():
            publisher = self.mask_pubs.get(camera)
            if not self._subscribed(publisher):
                continue
            shape = result.frame.depth_m.shape
            combined = np.zeros(shape, dtype=bool)
            for instance in result.instances:
                # Lazy-mask mode keeps the normal-frame mask on CUDA. This
                # D2H happens only after the ROS subscription check above, so
                # it cannot re-enter the numerical ScenePredictor hot path.
                combined |= instance_mask_cpu(instance, shape)
            array = np.ascontiguousarray(combined, dtype=np.uint8) * np.uint8(255)
            message = Image()
            message.header.stamp = _stamp_message(self.node, output.stamp_ns)
            message.header.frame_id = camera
            message.height, message.width = array.shape
            message.encoding = "mono8"
            message.is_bigendian = 0
            message.step = message.width
            message.data = array.tobytes()
            publisher.publish(message)

    @staticmethod
    def _exclusion_mask_cpu(result: Any, shape: tuple[int, int]) -> np.ndarray | None:
        """Materialize the per-view semantic exclusion union only on demand.

        Excluded EfficientTAM slots intentionally stop before 3D geometry and are
        represented downstream only by ``FrameResult.exclusion_mask_gpu``.  The
        annotated-RGB path is already subscriber/rate gated, so this single mask
        D2H stays completely outside the numerical hot path.
        """
        mask = getattr(result, "exclusion_mask_gpu", None)
        if mask is None:
            return None
        if torch.is_tensor(mask):
            value = mask.detach().cpu().numpy()
        else:
            value = np.asarray(mask)
        if tuple(value.shape) != tuple(shape):
            return None
        return np.asarray(value, dtype=bool)

    def _overlay_excluded(self, image: np.ndarray, result: Any) -> np.ndarray:
        """Overlay the exact dilated exclusion region used by rest-scene filtering."""
        mask = self._exclusion_mask_cpu(result, result.frame.depth_m.shape)
        if mask is None or not bool(mask.any()):
            return image

        color = _EXCLUDED_OVERLAY_COLOR
        blended = (
            image[mask].astype(np.float32) * 0.55
            + color.astype(np.float32) * 0.45
        )
        image[mask] = np.clip(blended, 0, 255).astype(np.uint8)

        labels = ", ".join(label for label, _ in self.config.tracker.excluded_prompts)
        text = f"Excluded: {labels}" if labels else "Excluded"
        cv2.putText(
            image,
            text,
            (8, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color.tolist(),
            2,
            cv2.LINE_AA,
        )
        return image

    def _publish_annotated_rgb(self, output: SceneVelocityOutput) -> None:
        for camera, result in output.view_results.items():
            publisher = self.annotated_rgb_pubs.get(camera)
            if not self._subscribed(publisher):
                continue
            # Both tracked and excluded overlays are intentionally materialized
            # only after the subscriber check. Excluded slots remain 2D-only.
            image = make_overlay(result)
            image = self._overlay_excluded(image, result)
            publisher.publish(_rgb8_image(self.node, image, output.stamp_ns, camera))

    def _velocity_lines(
        self,
        output: SceneVelocityOutput,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Drawn velocity lines: starts/ends [L, 3] float64 and their flow_points rows [L].

        Every velocity_marker_stride-th dense point with a finite velocity; the end
        point is point + velocity * flow_dt_s * velocity_marker_scale_factor.
        """
        stride = max(1, int(self.config.output.velocity_marker_stride))
        points = output.flow_points[::stride].detach().float().cpu().numpy()
        velocity = output.flow_velocity[::stride].detach().float().cpu().numpy()
        scale = float(output.flow_dt_s) * float(self.config.output.velocity_marker_scale_factor)
        rows = np.flatnonzero(
            np.isfinite(points).all(axis=1) & np.isfinite(velocity).all(axis=1)
        )
        starts = points[rows].astype(np.float64)
        ends = starts + scale * velocity[rows].astype(np.float64)
        return starts, ends, rows * stride

    def _publish_markers(
        self,
        output: SceneVelocityOutput,
        lines: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        array = MarkerArray()

        # Clear markers from the previous frame before adding the current set.
        delete = Marker()
        delete.header.stamp = _stamp_message(self.node, output.stamp_ns)
        delete.header.frame_id = self.config.ros.world_frame
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        # One LINE_LIST marker (segment i = points[2i], points[2i+1]) instead of
        # one ARROW marker per point: ~2000 Marker objects with headers/poses and
        # a clock call each took 10-100 ms of Python per publish.
        starts, ends, _ = lines
        segments = np.empty((2 * starts.shape[0], 3), dtype=np.float64)
        segments[0::2] = starts
        segments[1::2] = ends

        marker = Marker()
        marker.header.stamp = delete.header.stamp
        marker.header.frame_id = self.config.ros.world_frame
        marker.ns = "predicted_velocity"
        marker.id = 0
        marker.type = Marker.LINE_LIST
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        marker.scale.x = 0.003  # line width
        # Cyan velocity vectors (red is reserved for CBF critical points).
        marker.color.r = 0.1
        marker.color.g = 0.9
        marker.color.b = 1.0
        marker.color.a = 1.0
        marker.points = [Point(x=x, y=y, z=z) for x, y, z in segments.tolist()]
        array.markers.append(marker)
        self.marker_pub.publish(array)
        self._count_publish("velocity_markers")

    @staticmethod
    def _nearest_anchor_uncertainty(
        output: SceneVelocityOutput,
        rows: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Covariance [K, 3, 3], scale [K] and found flag [K] for flow_points[rows].

        Each dense point takes the uncertainty of its nearest warped anchor of the
        same track (any track when anchor ids are unavailable). Everything runs on
        the device of flow_points (other inputs are moved there), on the current
        (publisher) CUDA stream, for the selected points only; one D2H copy.
        """
        device = output.flow_points.device
        index = torch.as_tensor(rows, dtype=torch.int64, device=device)
        query = output.flow_points.index_select(0, index).float()
        anchors = output.warped_anchors.to(device, torch.float32)
        distance = torch.cdist(
            query.unsqueeze(0),
            anchors.unsqueeze(0),
            compute_mode="donot_use_mm_for_euclid_dist",
        )[0]
        anchor_ids = output.anchor_track_ids
        if anchor_ids is not None and anchor_ids.shape[0] == anchors.shape[0]:
            query_ids = output.flow_track_ids.to(device).index_select(0, index)
            other_track = query_ids.to(torch.int64).view(-1, 1) != anchor_ids.to(
                device, torch.int64
            ).view(1, -1)
            distance.masked_fill_(other_track, float("inf"))
        nearest_distance, nearest = distance.min(dim=1)
        packed = torch.cat(
            (
                output.anchor_flow_cov.to(device).index_select(0, nearest).reshape(-1, 9).float(),
                output.anchor_flow_cov_scale.to(device).index_select(0, nearest)
                .reshape(-1, 1).float(),
                torch.isfinite(nearest_distance).reshape(-1, 1).float(),
            ),
            dim=1,
        ).cpu().numpy().astype(np.float64)
        return packed[:, :9].reshape(-1, 3, 3), packed[:, 9], packed[:, 10] > 0.5

    def _publish_uncertainty_markers(
        self,
        output: SceneVelocityOutput,
        lines: tuple[np.ndarray, np.ndarray, np.ndarray],
    ) -> None:
        """Calibrated flow-uncertainty regions at the tips of (a subset of) the velocity lines.

        A drawn line's dense point takes Sigma and the calibration scale s of its
        nearest warped anchor of the same track. The region (uncertainty_region_markers)
        is centred at the line's end point p + dt * factor * v and scaled by the same
        factor = velocity_marker_scale_factor; with factor 1 its size is the anchor's
        calibrated one-frame region. Colour: uncertainty_colors of the one-frame
        equal-volume radius s det(Sigma)^(1/6), independent of factor.
        """
        output_config = self.config.output
        stamp = _stamp_message(self.node, output.stamp_ns)
        frame_id = self.config.ros.world_frame
        array = MarkerArray()

        # Clear the previous frame's markers first (also when nothing is drawn).
        delete = Marker()
        delete.header.stamp = stamp
        delete.header.frame_id = frame_id
        delete.ns = "flow_uncertainty"
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        _, ends, rows = lines
        covariance = output.anchor_flow_cov
        scale = output.anchor_flow_cov_scale
        if (
            output.flow_valid
            and covariance is not None
            and scale is not None
            and rows.size > 0
            and covariance.shape[0] > 0
            and covariance.shape[0] == output.warped_anchors.shape[0]
        ):
            # Marker count cap: per-marker Python cost stays bounded.
            selected = _evenly_spread(rows.size, int(output_config.uncertainty_marker_max))
            line_cov, line_scale, found = self._nearest_anchor_uncertainty(
                output, rows[selected]
            )
            keep = found & np.isfinite(line_cov).all(axis=(1, 2)) & np.isfinite(line_scale)
            positions, quaternions, diameters, radius = uncertainty_region_markers(
                self._uncertainty_region,
                ends[selected][keep],
                line_cov[keep],
                line_scale[keep],
                float(output_config.velocity_marker_scale_factor),
            )
            colors = uncertainty_colors(radius)
            header = Header(stamp=stamp, frame_id=frame_id)
            diameters = np.maximum(diameters, _MIN_MARKER_DIAMETER_M)
            pool = self._pooled_uncertainty_markers(positions.shape[0])
            for marker, position, quaternion, size, color in zip(
                pool, positions.tolist(), quaternions.tolist(), diameters.tolist(), colors.tolist()
            ):
                marker.header = header
                pose = marker.pose
                pose.position.x, pose.position.y, pose.position.z = position
                (
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ) = quaternion
                marker.scale.x, marker.scale.y, marker.scale.z = size
                marker.color.r, marker.color.g, marker.color.b = color
                array.markers.append(marker)
        self.uncertainty_marker_pub.publish(array)
        self._count_publish("flow_uncertainty_markers")

    def _pooled_uncertainty_markers(self, count: int) -> list[Marker]:
        """The first ``count`` pooled SPHERE markers (grown on demand). Ids start at 1: the
        DELETEALL of every message is (flow_uncertainty, 0), and RViz 2 drops a whole
        MarkerArray in which two markers share ns and id ("Duplicate Marker Check")."""
        pool = self._uncertainty_marker_pool
        alpha = float(self.config.output.uncertainty_marker_alpha)
        while len(pool) < count:
            marker = Marker()
            marker.ns = "flow_uncertainty"
            marker.id = len(pool) + 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.color.a = alpha
            pool.append(marker)
        return pool[:count]

    def _scene_cloud_due(self) -> bool:
        if self._scene_cloud_period_ns <= 0:
            return False
        due, self._next_scene_cloud_ns = _rate_gate(
            self._next_scene_cloud_ns, self._scene_cloud_period_ns, time.monotonic_ns()
        )
        return due

    def publish(self, output: SceneVelocityOutput) -> None:
        frame = self.config.ros.world_frame

        # Merged scene cloud copy: own (lower) rate, independent of the RViz gate.
        if (
            self.scene_cloud_pub is not None
            and output.scene_points is not None
            and self._scene_cloud_due()
            and self._subscribed(self.scene_cloud_pub)
        ):
            self.scene_cloud_pub.publish(
                _velocity_cloud(
                    self.node, output.scene_points, output.scene_velocity,
                    output.scene_track_ids, output.stamp_ns, frame,
                )
            )

        # Rate-limit before *any* subscriber query or expensive visualization work.
        # One decision gates every topic, keeping RGB/masks/PCDs/markers frame-aligned.
        if not self._visualization_publish_due():
            return

        if self._subscribed(self.tracked_pub):
            self.tracked_pub.publish(_tracked_cloud(self.node, output, frame))

        if self._subscribed(self.rest_scene_pub):
            if output.scene_points is not None:
                rest_points = output.scene_points[output.scene_num_dynamic:]
            else:
                rest_points = torch.empty((0, 3), dtype=torch.float32)
            self.rest_scene_pub.publish(
                _xyz_cloud(self.node, rest_points, output.stamp_ns, frame)
            )

        if self._subscribed(self.velocity_pub):
            self.velocity_pub.publish(
                _velocity_cloud(
                    self.node, output.flow_points, output.flow_velocity,
                    output.flow_track_ids, output.stamp_ns, frame,
                )
            )

        lines = None
        if self._subscribed(self.marker_pub):
            lines = self._velocity_lines(output)
            self._publish_markers(output, lines)

        if self._uncertainty_markers_due() and self._subscribed(self.uncertainty_marker_pub):
            if lines is None:
                lines = self._velocity_lines(output)
            self._publish_uncertainty_markers(output, lines)

        if self._subscribed(self.source_anchor_pub):
            self.source_anchor_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.source_anchors,
                    output.stamp_ns,
                    frame,
                    (80, 150, 240),
                )
            )
        if self._subscribed(self.warped_anchor_pub):
            self.warped_anchor_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.warped_anchors,
                    output.stamp_ns,
                    frame,
                    (235, 190, 70),
                )
            )

        if self._subscribed(self.outlier_removed_pub):
            self.outlier_removed_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.removed_outlier_points,
                    output.stamp_ns,
                    frame,
                    (170, 70, 220),
                )
            )

        if self.annotated_rgb_pubs:
            self._publish_annotated_rgb(output)
        if self.mask_pubs:
            self._publish_masks(output)
