from __future__ import annotations

import struct

import numpy as np
import torch
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from .config import PipelineConfig
from .data_types import SceneVelocityOutput


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
    """Publish tracker identities, dense scene velocity and flow diagnostics."""

    def __init__(self, node: Node, config: PipelineConfig) -> None:
        self.node = node
        self.config = config
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.tracked_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/tracked_points", qos
        )
        self.velocity_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/scene_velocity", qos
        )
        self.source_anchor_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/flow/source_anchors", qos
        )
        self.warped_anchor_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/flow/warped_anchors", qos
        )
        self.marker_pub = node.create_publisher(
            MarkerArray, "/scene_predictor/velocity_markers", qos
        )
        self.annotated_rgb_pubs = {
            camera: node.create_publisher(
                Image,
                f"/scene_predictor/{camera}/rgb_annotated",
                qos,
            )
            for camera in config.ros.camera_names
        }
        self.mask_pubs = {
            camera: node.create_publisher(
                Image,
                f"/scene_predictor/{camera}/tracked_mask",
                qos,
            )
            for camera in config.ros.camera_names
        }

    def _publish_masks(self, output: SceneVelocityOutput) -> None:
        for camera, mask in output.tracked_masks.items():
            publisher = self.mask_pubs.get(camera)
            if publisher is None:
                continue
            array = np.ascontiguousarray(mask, dtype=np.uint8) * np.uint8(255)
            message = Image()
            message.header.stamp = _stamp_message(self.node, output.stamp_ns)
            message.header.frame_id = camera
            message.height, message.width = array.shape
            message.encoding = "mono8"
            message.is_bigendian = 0
            message.step = message.width
            message.data = array.tobytes()
            publisher.publish(message)

    def _publish_markers(self, output: SceneVelocityOutput) -> None:
        stride = max(1, int(self.config.output.velocity_marker_stride))
        points = output.flow_points[::stride].detach().cpu().numpy()
        velocity = output.flow_velocity[::stride].detach().cpu().numpy()
        array = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        scale = float(self.config.output.velocity_marker_scale)
        for index, (point, vector) in enumerate(zip(points, velocity)):
            marker = Marker()
            marker.header.stamp = _stamp_message(self.node, output.stamp_ns)
            marker.header.frame_id = self.config.ros.world_frame
            marker.ns = "predicted_velocity"
            marker.id = index
            marker.type = Marker.ARROW
            marker.action = Marker.ADD
            marker.scale.x = 0.008
            marker.scale.y = 0.016
            marker.scale.z = 0.020
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
        frame = self.config.ros.world_frame
        if self.config.output.publish_tracked_objects:
            self.tracked_pub.publish(_tracked_cloud(self.node, output, frame))
        if self.config.output.publish_velocity_cloud:
            self.velocity_pub.publish(_velocity_cloud(self.node, output, frame))
        if self.config.output.publish_velocity_markers:
            self._publish_markers(output)
        if self.config.output.publish_flow_anchors:
            self.source_anchor_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.source_anchors,
                    output.stamp_ns,
                    frame,
                    (80, 150, 240),
                )
            )
            self.warped_anchor_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.warped_anchors,
                    output.stamp_ns,
                    frame,
                    (235, 190, 70),
                )
            )
        if self.config.output.publish_annotated_rgb:
            for camera, image in output.annotated_rgb.items():
                publisher = self.annotated_rgb_pubs.get(camera)
                if publisher is not None:
                    publisher.publish(
                        _rgb8_image(self.node, image, output.stamp_ns, camera)
                    )
        if self.config.output.publish_tracked_masks:
            self._publish_masks(output)
