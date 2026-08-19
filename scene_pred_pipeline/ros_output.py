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


_PALETTE = np.asarray(
    [
        [230, 80, 80],
        [80, 210, 120],
        [80, 150, 240],
        [235, 190, 70],
        [180, 90, 225],
        [70, 210, 215],
    ],
    dtype=np.uint8,
)


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
    output: SceneVelocityOutput,
    frame_id: str,
) -> PointCloud2:
    points = output.flow_points.detach().float().cpu().numpy()
    velocity = output.flow_velocity.detach().float().cpu().numpy()
    track_ids = output.flow_track_ids.detach().to(torch.int32).cpu().numpy()
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
    message.header = Header(stamp=_stamp_message(node, output.stamp_ns), frame_id=frame_id)
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
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
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
        self.source_anchor_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/flow/source_anchors", qos)
            if output.publish_flow_anchors else None
        )
        self.warped_anchor_pub = (
            node.create_publisher(PointCloud2, "/scene_predictor/flow/warped_anchors", qos)
            if output.publish_flow_anchors else None
        )
        self.marker_pub = (
            node.create_publisher(MarkerArray, "/scene_predictor/velocity_markers", qos)
            if output.publish_velocity_markers else None
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

        # Rest-scene geometry is a visualization-only CUDA path.  All persistent
        # setup is tiny and happens once; no per-frame work is submitted unless an
        # actual subscriber is connected to /scene_predictor/rest_points.
        self._rest_ray_cache: dict[
            tuple[Any, ...], tuple[torch.Tensor, torch.Tensor]
        ] = {}
        self._rest_device = torch.device(config.runtime.device)
        if self._rest_device.type == "cuda" and self._rest_device.index is None:
            self._rest_device = torch.device("cuda", torch.cuda.current_device())
        self._rest_voxel_size_m = 0.0
        self._rest_inv_voxel_size = 0.0
        self._rest_origin = None
        self._rest_min_depth_m = 0.0
        self._rest_max_depth_m = float("inf")
        if self.rest_scene_pub is not None:
            if tracker_config is None:
                # Backward-compatible fallback for external RosVisualizer users.
                # ScenePredictor itself passes the already-loaded native config so
                # its tracker and visualization always share the exact same values.
                from sam_rgbd_tracking.config import load_config as load_tracker_config

                tracker_config = load_tracker_config(config.tracker.config_path)
            self._configure_rest_scene(tracker_config)

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

        now_ns = time.monotonic_ns()
        if now_ns < self._next_visualization_ns:
            return False

        self._next_visualization_ns = now_ns + period_ns
        return True

    def _configure_rest_scene(self, tracker_config: Any) -> None:
        try:
            voxel_size_m = float(tracker_config.shared_voxel_grid.voxel_size_m)
            origin_world = np.asarray(
                tracker_config.shared_voxel_grid.origin_world, dtype=np.float32
            ).reshape(3)
            min_depth_m = float(tracker_config.postprocess.min_valid_depth_m)
            max_depth_m = float(tracker_config.postprocess.max_valid_depth_m)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "Rest-scene visualization requires tracker "
                "shared_voxel_grid.voxel_size_m/origin_world and "
                "postprocess min/max depth settings."
            ) from exc
        if voxel_size_m <= 0.0:
            raise ValueError(
                f"shared_voxel_grid.voxel_size_m must be positive, got {voxel_size_m}"
            )
        if max_depth_m < min_depth_m:
            raise ValueError(
                "postprocess.max_valid_depth_m must be >= min_valid_depth_m"
            )

        self._rest_voxel_size_m = voxel_size_m
        self._rest_inv_voxel_size = 1.0 / voxel_size_m
        self._rest_origin = torch.as_tensor(
            origin_world, dtype=torch.float32, device=self._rest_device
        ).contiguous()
        self._rest_min_depth_m = min_depth_m
        self._rest_max_depth_m = max_depth_m

    def _rest_rays(self, frame: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Return cached flattened camera rays for dense GPU backprojection."""
        intrinsics = frame.intrinsics
        height, width = map(int, frame.depth_m.shape)
        key = (
            height,
            width,
            float(intrinsics.fx),
            float(intrinsics.fy),
            float(intrinsics.cx),
            float(intrinsics.cy),
            str(self._rest_device),
        )
        cached = self._rest_ray_cache.get(key)
        if cached is not None:
            return cached

        x_ray = (
            torch.arange(width, dtype=torch.float32, device=self._rest_device)
            - float(intrinsics.cx)
        ) / float(intrinsics.fx)
        y_ray = (
            torch.arange(height, dtype=torch.float32, device=self._rest_device)
            - float(intrinsics.cy)
        ) / float(intrinsics.fy)
        # Flattened dense rays avoid torch.nonzero() and its CUDA host sync.  The
        # fixed-size dense arithmetic is cheap on the GPU and lets us sort one
        # shared voxel-key array for all cameras.
        cached = (x_ray.repeat(height), y_ray.repeat_interleave(width))
        self._rest_ray_cache[key] = cached
        return cached

    def _rest_exclusion_mask_gpu(
        self, result: Any, shape: tuple[int, int]
    ) -> torch.Tensor:
        """Union tracked final masks + the tracker-provided exclusion-only mask."""
        excluded_cpu = np.zeros(shape, dtype=bool)
        gpu_masks: list[torch.Tensor] = []
        for instance in result.instances:
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

        excluded = torch.as_tensor(
            excluded_cpu, dtype=torch.bool, device=self._rest_device
        )
        for mask_gpu in gpu_masks:
            if mask_gpu.device != self._rest_device:
                mask_gpu = mask_gpu.to(self._rest_device)
            if mask_gpu.dtype == torch.bool:
                excluded.logical_or_(mask_gpu)
            else:
                excluded.logical_or_(mask_gpu != 0)
        return excluded

    def _rest_scene_points(self, output: SceneVelocityOutput) -> torch.Tensor:
        """Build one white rest-scene cloud on the tracker's exact world lattice.

        The path stays on CUDA through mask exclusion, depth validation, dense
        backprojection, world transform, voxel quantization and cross-view unique.
        Only the final compact XYZ cloud is copied to CPU by _xyz_cloud().
        """
        if self._rest_origin is None:
            return torch.empty(
                (0, 3), dtype=torch.float32, device=self._rest_device
            )

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

            depth = torch.as_tensor(
                depth_cpu, dtype=torch.float32, device=self._rest_device
            ).reshape(-1)
            excluded = self._rest_exclusion_mask_gpu(result, shape).reshape(-1)
            valid = (
                (~excluded)
                & torch.isfinite(depth)
                & (depth >= self._rest_min_depth_m)
                & (depth <= self._rest_max_depth_m)
            )

            # Keep a fixed H*W shape instead of CUDA nonzero()/variable-size
            # compaction. Invalid pixels use z=0 and receive a sentinel voxel key.
            z = depth.clone()
            z.masked_fill_(~valid, 0.0)
            ray_x, ray_y = self._rest_rays(frame)
            points_camera = torch.empty(
                (height * width, 3),
                dtype=torch.float32,
                device=self._rest_device,
            )
            points_camera[:, 0] = ray_x * z
            points_camera[:, 1] = ray_y * z
            points_camera[:, 2] = z

            transform = torch.as_tensor(
                transform_cpu, dtype=torch.float32, device=self._rest_device
            )
            points_world = points_camera @ transform[:3, :3].T
            points_world.add_(transform[:3, 3])

            voxel_coords = torch.floor(
                (points_world - self._rest_origin) * self._rest_inv_voxel_size
            ).to(torch.int64)
            shifted = voxel_coords + bias
            in_key_range = (shifted >= 0).all(dim=1) & (shifted <= key_mask).all(dim=1)
            valid.logical_and_(in_key_range)

            keys = (
                (shifted[:, 0] << 42)
                | (shifted[:, 1] << 21)
                | shifted[:, 2]
            )
            keys.masked_fill_(~valid, invalid_key)
            all_points.append(points_world)
            all_keys.append(keys)

        if not all_points:
            return torch.empty(
                (0, 3), dtype=torch.float32, device=self._rest_device
            )

        points = all_points[0] if len(all_points) == 1 else torch.cat(all_points, dim=0)
        keys = all_keys[0] if len(all_keys) == 1 else torch.cat(all_keys, dim=0)

        # One global sort deduplicates both each camera and cross-view overlap on
        # exactly the same voxel_size/origin lattice used by selected objects.
        sorted_keys, order = torch.sort(keys)
        keep = torch.empty_like(sorted_keys, dtype=torch.bool)
        keep[0] = sorted_keys[0] != invalid_key
        if sorted_keys.numel() > 1:
            keep[1:] = (sorted_keys[1:] != sorted_keys[:-1]) & (
                sorted_keys[1:] != invalid_key
            )
        return points[order[keep]]

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

    def _publish_markers(self, output: SceneVelocityOutput) -> None:
        stride = max(1, int(self.config.output.velocity_marker_stride))
        points = output.flow_points[::stride].detach().float().cpu().numpy()
        velocity = output.flow_velocity[::stride].detach().float().cpu().numpy()
        array = MarkerArray()

        # Clear markers from the previous frame before adding the current set.
        delete = Marker()
        delete.header.stamp = _stamp_message(self.node, output.stamp_ns)
        delete.header.frame_id = self.config.ros.world_frame
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        scale = float(self.config.output.velocity_marker_scale)
        for index, (point, vector) in enumerate(zip(points, velocity)):
            if not (np.isfinite(point).all() and np.isfinite(vector).all()):
                continue

            marker = Marker()
            marker.header.stamp = _stamp_message(self.node, output.stamp_ns)
            marker.header.frame_id = self.config.ros.world_frame
            marker.ns = "predicted_velocity"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.pose.orientation.x = 0.0
            marker.pose.orientation.y = 0.0
            marker.pose.orientation.z = 0.0
            marker.pose.orientation.w = 1.0
            marker.scale.x = 0.003
            marker.scale.y = 0.006
            marker.scale.z = 0.008
            marker.color.r = 1.0
            marker.color.g = 0.25
            marker.color.b = 0.05
            marker.color.a = 1.0
            marker.points = [
                Point(x=float(point[0]), y=float(point[1]), z=float(point[2])),
                Point(
                    x=float(point[0] + scale * vector[0]),
                    y=float(point[1] + scale * vector[1]),
                    z=float(point[2] + scale * vector[2]),
                ),
            ]
            array.markers.append(marker)
        self.marker_pub.publish(array)

    def publish(self, output: SceneVelocityOutput) -> None:
        # Rate-limit before *any* subscriber query or expensive visualization work.
        # One decision gates every topic, keeping RGB/masks/PCDs/markers frame-aligned.
        if not self._visualization_publish_due():
            return

        frame = self.config.ros.world_frame

        if self._subscribed(self.tracked_pub):
            self.tracked_pub.publish(_tracked_cloud(self.node, output, frame))

        if self._subscribed(self.rest_scene_pub):
            rest_points = self._rest_scene_points(output)
            self.rest_scene_pub.publish(
                _xyz_cloud(self.node, rest_points, output.stamp_ns, frame)
            )

        if self._subscribed(self.velocity_pub):
            self.velocity_pub.publish(_velocity_cloud(self.node, output, frame))

        if self._subscribed(self.marker_pub):
            self._publish_markers(output)

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

        if self.annotated_rgb_pubs:
            self._publish_annotated_rgb(output)
        if self.mask_pubs:
            self._publish_masks(output)
