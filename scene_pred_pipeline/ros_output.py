from __future__ import annotations

import struct

import numpy as np
import torch
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.qos import (
    HistoryPolicy,
    QoSProfile,
    ReliabilityPolicy,
)
from sensor_msgs.msg import (
    Image,
    PointCloud2,
    PointField,
)
from std_msgs.msg import Header
from visualization_msgs.msg import (
    Marker,
    MarkerArray,
)

from .config import PipelineConfig
from .data_types import SceneVelocityOutput


def _stamp_message(
    node: Node,
    stamp_ns: int,
):
    message = node.get_clock().now().to_msg()
    message.sec = (
        stamp_ns // 1_000_000_000
    )
    message.nanosec = (
        stamp_ns % 1_000_000_000
    )
    return message


def _rgb_float(
    red: int,
    green: int,
    blue: int,
) -> float:
    packed = (
        red << 16
    ) | (
        green << 8
    ) | blue
    return struct.unpack(
        "f",
        struct.pack("I", packed),
    )[0]


def _xyzrgb_cloud(
    node: Node,
    points: torch.Tensor,
    stamp_ns: int,
    frame_id: str,
    rgb: tuple[int, int, int],
) -> PointCloud2:
    cpu = (
        points.detach()
        .float()
        .cpu()
        .numpy()
    )
    packed = np.full(
        (cpu.shape[0], 1),
        _rgb_float(*rgb),
        dtype=np.float32,
    )
    cloud = np.ascontiguousarray(
        np.concatenate(
            (cpu, packed),
            axis=1,
        )
    )

    message = PointCloud2()
    message.header = Header(
        stamp=_stamp_message(
            node,
            stamp_ns,
        ),
        frame_id=frame_id,
    )
    message.height = 1
    message.width = cloud.shape[0]
    message.fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="rgb",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
    ]
    message.is_bigendian = False
    message.point_step = 16
    message.row_step = (
        16 * cloud.shape[0]
    )
    message.data = cloud.tobytes()
    message.is_dense = True
    return message


def _velocity_cloud(
    node: Node,
    output: SceneVelocityOutput,
    frame_id: str,
) -> PointCloud2:
    points = (
        output.moving_points
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    velocity = (
        output.moving_velocity
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    track_ids = (
        output.moving_track_ids
        .detach()
        .to(torch.int32)
        .cpu()
        .numpy()
    )

    structured = np.empty(
        points.shape[0],
        dtype=[
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("vx", "<f4"),
            ("vy", "<f4"),
            ("vz", "<f4"),
            ("track_id", "<i4"),
        ],
    )
    if points.shape[0] > 0:
        (
            structured["x"],
            structured["y"],
            structured["z"],
        ) = points.T
        (
            structured["vx"],
            structured["vy"],
            structured["vz"],
        ) = velocity.T
        structured["track_id"] = track_ids

    message = PointCloud2()
    message.header = Header(
        stamp=_stamp_message(
            node,
            output.stamp_ns,
        ),
        frame_id=frame_id,
    )
    message.height = 1
    message.width = points.shape[0]
    message.fields = [
        PointField(
            name="x",
            offset=0,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="y",
            offset=4,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="z",
            offset=8,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="vx",
            offset=12,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="vy",
            offset=16,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="vz",
            offset=20,
            datatype=PointField.FLOAT32,
            count=1,
        ),
        PointField(
            name="track_id",
            offset=24,
            datatype=PointField.INT32,
            count=1,
        ),
    ]
    message.is_bigendian = False
    message.point_step = (
        structured.dtype.itemsize
    )
    message.row_step = (
        message.point_step
        * points.shape[0]
    )
    message.data = structured.tobytes()
    message.is_dense = True
    return message


def _rgb8_image(
    node: Node,
    image: np.ndarray,
    stamp_ns: int,
    frame_id: str,
) -> Image:
    image = np.ascontiguousarray(
        image,
        dtype=np.uint8,
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


class RosVisualizer:
    """Optional RViz publishing.

    If runtime.enable_visualization is false, no publisher is created and
    publish() returns immediately. This avoids all GPU->CPU visualization
    transfers.
    """

    def __init__(
        self,
        node: Node,
        config: PipelineConfig,
    ) -> None:
        self.node = node
        self.config = config
        self.enabled = bool(
            config.runtime.enable_visualization
        )
        if not self.enabled:
            return

        cloud_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        image_qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
        )

        self.background_pub = (
            node.create_publisher(
                PointCloud2,
                "/scene_predictor/background_points",
                cloud_qos,
            )
        )
        self.static_pub = node.create_publisher(
            PointCloud2,
            "/scene_predictor/static_object_points",
            cloud_qos,
        )
        self.moving_pub = node.create_publisher(
            PointCloud2,
            "/scene_predictor/moving_object_points",
            cloud_qos,
        )
        self.velocity_pub = node.create_publisher(
            PointCloud2,
            "/scene_predictor/scene_velocity",
            cloud_qos,
        )
        self.marker_pub = node.create_publisher(
            MarkerArray,
            "/scene_predictor/velocity_markers",
            cloud_qos,
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
                (
                    f"/scene_predictor/"
                    f"{camera}/moving_mask"
                ),
                image_qos,
            )
            for camera in config.ros.camera_names
        }

    def _publish_masks(
        self,
        output: SceneVelocityOutput,
    ) -> None:
        for camera, mask in (
            output.moving_masks.items()
        ):
            publisher = self.mask_pubs.get(camera)
            if publisher is None:
                continue
            array = (
                mask.detach()
                .to(torch.uint8)
                .mul_(255)
                .cpu()
                .numpy()
            )
            message = Image()
            message.header.stamp = _stamp_message(
                self.node,
                output.stamp_ns,
            )
            message.header.frame_id = camera
            (
                message.height,
                message.width,
            ) = array.shape
            message.encoding = "mono8"
            message.is_bigendian = 0
            message.step = message.width
            message.data = (
                np.ascontiguousarray(
                    array
                ).tobytes()
            )
            publisher.publish(message)

    def _publish_markers(
        self,
        output: SceneVelocityOutput,
    ) -> None:
        stride = max(
            1,
            self.config.output.velocity_marker_stride,
        )
        points = (
            output.moving_points[::stride]
            .detach()
            .cpu()
            .numpy()
        )
        velocity = (
            output.moving_velocity[::stride]
            .detach()
            .cpu()
            .numpy()
        )

        array = MarkerArray()
        delete = Marker()
        delete.action = Marker.DELETEALL
        array.markers.append(delete)

        scale = (
            self.config.output.velocity_marker_scale
        )
        for index, (
            point,
            vector,
        ) in enumerate(
            zip(points, velocity)
        ):
            marker = Marker()
            marker.header.stamp = _stamp_message(
                self.node,
                output.stamp_ns,
            )
            marker.header.frame_id = (
                self.config.ros.world_frame
            )
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

            start = Point(
                x=float(point[0]),
                y=float(point[1]),
                z=float(point[2]),
            )
            end_point = (
                point + scale * vector
            )
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
        for camera_name, image in (
            output.annotated_rgb.items()
        ):
            publisher = (
                self.annotated_rgb_pubs.get(
                    camera_name
                )
            )
            if publisher is None:
                continue
            publisher.publish(
                _rgb8_image(
                    self.node,
                    image,
                    output.stamp_ns,
                    camera_name,
                )
            )

    def publish(
        self,
        output: SceneVelocityOutput,
    ) -> None:
        if not self.enabled:
            return

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
        if (
            self.config.output.publish_static_objects
        ):
            self.static_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.static_points,
                    output.stamp_ns,
                    frame,
                    (40, 220, 80),
                )
            )
        if (
            self.config.output.publish_moving_objects
        ):
            self.moving_pub.publish(
                _xyzrgb_cloud(
                    self.node,
                    output.moving_points,
                    output.stamp_ns,
                    frame,
                    (240, 50, 30),
                )
            )
        if (
            self.config.output.publish_velocity_cloud
        ):
            self.velocity_pub.publish(
                _velocity_cloud(
                    self.node,
                    output,
                    frame,
                )
            )
        if (
            self.config.output.publish_velocity_markers
        ):
            self._publish_markers(output)
        if (
            self.config.output.publish_annotated_rgb
        ):
            self._publish_annotated_rgb(output)
        if (
            self.config.output.publish_moving_masks
        ):
            self._publish_masks(output)
