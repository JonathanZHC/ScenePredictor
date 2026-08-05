from __future__ import annotations

import struct

import numpy as np
import torch
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray
import cv2

from .config import PipelineConfig
from .data_types import (
    MotionState,
    SceneVelocityOutput,
)


def _stamp_message(node: Node, stamp_ns: int):
    message = node.get_clock().now().to_msg()
    message.sec = stamp_ns // 1_000_000_000
    message.nanosec = stamp_ns % 1_000_000_000
    return message


def _rgb_float(red: int, green: int, blue: int) -> float:
    packed = (red << 16) | (green << 8) | blue
    return struct.unpack("f", struct.pack("I", packed))[0]


def _xyzrgb_cloud(
    node: Node,
    points: torch.Tensor,
    stamp_ns: int,
    frame_id: str,
    rgb: tuple[int, int, int],
) -> PointCloud2:
    cpu = points.detach().float().cpu().numpy()
    packed = np.full(
        (cpu.shape[0], 1),
        _rgb_float(*rgb),
        dtype=np.float32,
    )
    cloud = np.ascontiguousarray(np.concatenate((cpu, packed), axis=1))
    message = PointCloud2()
    message.header = Header(
        stamp=_stamp_message(node, stamp_ns),
        frame_id=frame_id,
    )
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


def _velocity_cloud(
    node: Node,
    output: SceneVelocityOutput,
    frame_id: str,
) -> PointCloud2:
    points = output.moving_points.detach().float().cpu().numpy()
    velocity = output.moving_velocity.detach().float().cpu().numpy()
    track_ids = output.moving_track_ids.detach().to(torch.int32).cpu().numpy()
    structured = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("vx", "<f4"), ("vy", "<f4"), ("vz", "<f4"),
            ("track_id", "<i4"),
        ],
    )
    structured["x"], structured["y"], structured["z"] = points.T
    structured["vx"], structured["vy"], structured["vz"] = velocity.T
    structured["track_id"] = track_ids
    message = PointCloud2()
    message.header = Header(
        stamp=_stamp_message(node, output.stamp_ns),
        frame_id=frame_id,
    )
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


def _rgb8_message(
    node: Node,
    image: np.ndarray,
    stamp_ns: int,
    frame_id: str,
) -> Image:
    image = np.ascontiguousarray(
        image,
        dtype=np.uint8,
    )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 RGB image, got "
            f"{image.shape}."
        )

    message = Image()
    message.header.stamp = _stamp_message(
        node,
        stamp_ns,
    )
    message.header.frame_id = frame_id

    message.height = image.shape[0]
    message.width = image.shape[1]
    message.encoding = "rgb8"
    message.is_bigendian = 0
    message.step = image.shape[1] * 3
    message.data = image.tobytes()

    return message


def _annotated_rgb(
    rgb: np.ndarray,
    detections,
) -> np.ndarray:
    image = np.ascontiguousarray(
        rgb.copy(),
        dtype=np.uint8,
    )

    height, width = image.shape[:2]

    for detection in detections:
        x0, y0, x1, y1 = detection.bbox_xyxy

        x0 = max(0, min(width - 1, x0))
        x1 = max(0, min(width - 1, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(0, min(height - 1, y1))

        is_moving = (
            detection.motion_state
            == MotionState.MOVING
        )

        # Image is RGB, not BGR.
        color = (
            (255, 60, 40)
            if is_moving
            else (40, 220, 80)
        )

        state = (
            "MOVING"
            if is_moving
            else "STATIC"
        )

        label = (
            f"{detection.class_name} "
            f"{detection.confidence:.2f} "
            f"id={detection.track_id} "
            f"{state}"
        )

        cv2.rectangle(
            image,
            (x0, y0),
            (x1, y1),
            color,
            thickness=2,
            lineType=cv2.LINE_AA,
        )

        (
            text_width,
            text_height,
        ), baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            1,
        )

        text_y = max(
            text_height + baseline + 2,
            y0,
        )

        cv2.rectangle(
            image,
            (
                x0,
                text_y - text_height - baseline - 4,
            ),
            (
                min(width - 1, x0 + text_width + 4),
                text_y + 2,
            ),
            color,
            thickness=-1,
        )

        cv2.putText(
            image,
            label,
            (x0 + 2, text_y - baseline),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            thickness=1,
            lineType=cv2.LINE_AA,
        )

    return image


class RosVisualizer:
    """Publish masks, classified points and recovered velocity for RViz."""

    def __init__(self, node: Node, config: PipelineConfig) -> None:
        self.node = node
        self.config = config
        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        self.background_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/background_points", qos
        )
        self.static_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/static_object_points", qos
        )
        self.moving_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/moving_object_points", qos
        )
        self.velocity_pub = node.create_publisher(
            PointCloud2, "/scene_predictor/scene_velocity", qos
        )
        self.marker_pub = node.create_publisher(
            MarkerArray, "/scene_predictor/velocity_markers", qos
        )
        self.annotated_rgb_pubs = {
            camera: node.create_publisher(
                Image,
                (
                    f"/scene_predictor/"
                    f"{camera}/rgb_annotated"
                ),
                image_qos,
            )
            for camera in config.ros.camera_names
        }
        self.mask_pubs = {
            camera: node.create_publisher(
                Image,
                f"/scene_predictor/{camera}/moving_mask",
                qos,
            )
            for camera in config.ros.camera_names
        }

    def _publish_masks(self, output: SceneVelocityOutput) -> None:
        for camera, mask in output.moving_masks.items():
            array = (
                mask.detach().to(torch.uint8).mul_(255).cpu().numpy()
            )
            message = Image()
            message.header.stamp = _stamp_message(self.node, output.stamp_ns)
            message.header.frame_id = camera
            message.height, message.width = array.shape
            message.encoding = "mono8"
            message.is_bigendian = 0
            message.step = message.width
            message.data = np.ascontiguousarray(array).tobytes()
            self.mask_pubs[camera].publish(message)

    def _publish_markers(self, output: SceneVelocityOutput) -> None:
        stride = max(1, self.config.output.velocity_marker_stride)
        points = output.moving_points[::stride].detach().cpu().numpy()
        velocity = output.moving_velocity[::stride].detach().cpu().numpy()
        array = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        scale = self.config.output.velocity_marker_scale
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
            start = Point(x=float(point[0]), y=float(point[1]), z=float(point[2]))
            end_point = point + scale * vector
            end = Point(
                x=float(end_point[0]),
                y=float(end_point[1]),
                z=float(end_point[2]),
            )
            marker.points = [start, end]
            array.markers.append(marker)
        self.marker_pub.publish(array)
    
    def _publish_annotated_rgb(
        self,
        output: SceneVelocityOutput,
    ) -> None:
        for camera_name, rgb in output.camera_rgb.items():
            annotated = _annotated_rgb(
                rgb,
                output.image_detections.get(
                    camera_name,
                    [],
                ),
            )

            self.annotated_rgb_pubs[
                camera_name
            ].publish(
                _rgb8_message(
                    self.node,
                    annotated,
                    output.stamp_ns,
                    camera_name,
                )
            )

    def publish(self, output: SceneVelocityOutput) -> None:
        frame = self.config.ros.world_frame
        if self.config.output.publish_background:
            self.background_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.background_points,
                    output.stamp_ns,
                    frame,
                    (128, 128, 128),
                )
            )
        if self.config.output.publish_static_objects:
            self.static_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.static_points,
                    output.stamp_ns,
                    frame,
                    (40, 220, 80),
                )
            )
        if self.config.output.publish_moving_objects:
            self.moving_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.moving_points,
                    output.stamp_ns,
                    frame,
                    (240, 50, 30),
                )
            )
        if self.config.output.publish_velocity_cloud:
            self.velocity_pub.publish(
                _velocity_cloud(self.node, output, frame)
            )
        if self.config.output.publish_velocity_markers:
            self._publish_markers(output)
        if self.config.output.publish_annotated_rgb:
            self._publish_annotated_rgb(output)
        if self.config.output.publish_moving_masks:
            self._publish_masks(output)
