from __future__ import annotations

from collections import defaultdict
import threading
import time
from typing import Callable

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CameraInfo, Image

from .config import PipelineConfig
from .data_types import CameraFrameCpu, MultiCameraFrame


def _stamp_ns(message) -> int:
    return (
        int(message.header.stamp.sec) * 1_000_000_000
        + int(message.header.stamp.nanosec)
    )


def _image_to_numpy(message: Image) -> np.ndarray:
    dtype_map = {
        "rgb8": np.uint8,
        "32FC1": np.float32,
    }
    dtype = dtype_map.get(message.encoding)
    if dtype is None:
        raise ValueError(f"Unsupported image encoding: {message.encoding}")
    channels = 3 if message.encoding == "rgb8" else 1
    array = np.frombuffer(message.data, dtype=dtype)
    expected = message.height * message.width * channels
    if array.size != expected:
        raise ValueError(
            f"Image payload has {array.size} values, expected {expected}."
        )
    shape = (
        (message.height, message.width, channels)
        if channels > 1
        else (message.height, message.width)
    )
    return np.ascontiguousarray(array.reshape(shape))


def _pose_matrix(message: PoseStamped) -> np.ndarray:
    q = message.pose.orientation
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    norm = max(1.0e-12, np.sqrt(x*x + y*y + z*z + w*w))
    x, y, z, w = x/norm, y/norm, z/norm, w/norm
    rotation = np.array(
        [
            [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
            [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
            [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
        ],
        dtype=np.float32,
    )
    output = np.eye(4, dtype=np.float32)
    output[:3, :3] = rotation
    output[:3, 3] = (
        message.pose.position.x,
        message.pose.position.y,
        message.pose.position.z,
    )
    return output


class MultiCameraRosInput:
    """Dynamic-camera ROS input using exact publisher timestamps."""

    def __init__(
        self,
        node: Node,
        config: PipelineConfig,
        callback: Callable[[MultiCameraFrame], None],
    ) -> None:
        self.node = node
        self.config = config
        self.callback = callback
        self.lock = threading.Lock()
        self.pending: dict[int, dict[str, dict[str, object]]] = defaultdict(
            lambda: defaultdict(dict)
        )
        self.first_seen: dict[int, float] = {}
        self.subscriptions = []

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=config.ros.queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        for camera in config.ros.camera_names:
            bindings = {
                "rgb": (
                    Image,
                    config.ros.color_topic.format(camera=camera),
                ),
                "depth": (
                    Image,
                    config.ros.depth_topic.format(camera=camera),
                ),
                "info": (
                    CameraInfo,
                    config.ros.camera_info_topic.format(camera=camera),
                ),
                "pose": (
                    PoseStamped,
                    config.ros.pose_topic.format(camera=camera),
                ),
            }
            for kind, (message_type, topic) in bindings.items():
                subscription = node.create_subscription(
                    message_type,
                    topic,
                    lambda message, c=camera, k=kind: self._receive(c, k, message),
                    qos,
                )
                self.subscriptions.append(subscription)

    def _receive(self, camera: str, kind: str, message) -> None:
        stamp = _stamp_ns(message)
        with self.lock:
            self.pending[stamp][camera][kind] = message
            self.first_seen.setdefault(stamp, time.monotonic())
            ready = self._try_build(stamp)
            self._drop_expired()
        if ready is not None:
            self.callback(ready)

    def _try_build(self, stamp: int) -> MultiCameraFrame | None:
        camera_data = self.pending[stamp]
        required = {"rgb", "depth", "info", "pose"}
        if not all(
            required.issubset(camera_data.get(camera, {}))
            for camera in self.config.ros.camera_names
        ):
            return None

        frames: dict[str, CameraFrameCpu] = {}
        for camera in self.config.ros.camera_names:
            values = camera_data[camera]
            info: CameraInfo = values["info"]
            frames[camera] = CameraFrameCpu(
                camera_name=camera,
                stamp_ns=stamp,
                rgb=_image_to_numpy(values["rgb"]),
                depth=_image_to_numpy(values["depth"]),
                K=np.asarray(info.k, dtype=np.float32).reshape(3, 3),
                T_world_camera=_pose_matrix(values["pose"]),
                optical_frame_id=values["rgb"].header.frame_id,
            )

        del self.pending[stamp]
        self.first_seen.pop(stamp, None)
        return MultiCameraFrame(stamp_ns=stamp, cameras=frames)

    def _drop_expired(self) -> None:
        timeout_s = self.config.ros.incomplete_timeout_ms / 1000.0
        now = time.monotonic()
        expired = [
            stamp
            for stamp, start in self.first_seen.items()
            if now - start > timeout_s
        ]
        for stamp in expired:
            self.pending.pop(stamp, None)
            self.first_seen.pop(stamp, None)
