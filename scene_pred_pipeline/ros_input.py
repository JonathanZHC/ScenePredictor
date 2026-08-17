from __future__ import annotations

from dataclasses import dataclass
import threading
import time
from typing import Callable

import numpy as np
import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import Buffer, TransformListener

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
        "16UC1": np.uint16,
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


def _depth_to_meters(message: Image) -> np.ndarray:
    depth = _image_to_numpy(message)
    if message.encoding == "16UC1":
        # The current real-camera path uses uint16 millimetres.
        return np.ascontiguousarray(depth, dtype=np.float32) * 0.001
    if message.encoding == "32FC1":
        return np.ascontiguousarray(depth, dtype=np.float32)
    raise ValueError(f"Unsupported depth encoding: {message.encoding}")


def _transform_matrix(message) -> np.ndarray:
    transform = message.transform
    q = transform.rotation
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
        float(transform.translation.x),
        float(transform.translation.y),
        float(transform.translation.z),
    )
    return output


@dataclass(frozen=True)
class _RgbDepthPair:
    """One camera-local RGB-D pair; RGB time is the canonical view time."""

    stamp_ns: int
    rgb: Image
    depth: Image


class MultiCameraRosInput:
    """Approximate RGB-D input with cached intrinsics and timestamped TF.

    Each camera first pairs RGB with the nearest depth frame within
    ``sync_slop_seconds``. CameraInfo is cached independently because camera
    calibration is not a per-frame synchronization signal. For multiple
    cameras, completed camera-local pairs are then aligned to camera_names[0]
    within ``multiview_sync_slop_seconds``.
    """

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
        self.camera_names = tuple(str(name) for name in config.ros.camera_names)
        self.sync_slop_ns = int(round(float(config.ros.sync_slop_seconds) * 1.0e9))
        self.multiview_slop_ns = int(
            round(float(config.ros.multiview_sync_slop_seconds) * 1.0e9)
        )
        self.sync_queue_size = max(2, int(config.ros.queue_depth))

        self.rgb_pending: dict[str, list[tuple[int, Image]]] = {
            camera: [] for camera in self.camera_names
        }
        self.depth_pending: dict[str, list[tuple[int, Image]]] = {
            camera: [] for camera in self.camera_names
        }
        self.local_pairs: dict[str, list[_RgbDepthPair]] = {
            camera: [] for camera in self.camera_names
        }
        self.camera_info: dict[str, CameraInfo] = {}

        # ScenePredictor is latest-only. Never allow late ROS delivery to move
        # the stateful tracker / scene-flow pipeline backward in time.
        self.last_emitted_stamp_ns: int | None = None
        self.subscriptions = []
        self._last_tf_warning_s: dict[str, float] = {}

        # The main ScenePredictor node is spun by rclpy.spin() in a single
        # executor thread. Use a dedicated TF node/thread so a timestamped
        # lookup can wait briefly for /tf without blocking TF callbacks.
        self.tf_node = rclpy.create_node(f"{node.get_name()}_tf_listener")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer,
            self.tf_node,
            spin_thread=True,
        )

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=config.ros.queue_depth,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )
        for camera in self.camera_names:
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
        candidate: dict[str, _RgbDepthPair] | None
        with self.lock:
            if kind == "info":
                # Intrinsics are effectively static for one active stream
                # profile. Do not require CameraInfo to share an exact stamp
                # with every RGB-D pair.
                self.camera_info[camera] = message
            else:
                stamp_ns = _stamp_ns(message)
                queue = (
                    self.rgb_pending[camera]
                    if kind == "rgb"
                    else self.depth_pending[camera]
                )
                queue.append((stamp_ns, message))
                queue.sort(key=lambda item: item[0])
                if len(queue) > self.sync_queue_size:
                    del queue[: len(queue) - self.sync_queue_size]
                self._pair_camera_locked(camera)

            candidate = self._select_multiview_candidate_locked()

        if candidate is None:
            return

        ready = self._build_frame(candidate)
        if ready is None:
            # Keep the candidate queued. A later image/info callback retries the
            # timestamped TF lookup; this matters when TF arrives just after data.
            return

        with self.lock:
            if not self._candidate_is_current_locked(candidate):
                return
            self._consume_candidate_locked(candidate)
            if (
                self.last_emitted_stamp_ns is not None
                and ready.stamp_ns <= self.last_emitted_stamp_ns
            ):
                return
            self.last_emitted_stamp_ns = ready.stamp_ns

        self.callback(ready)

    def _pair_camera_locked(self, camera: str) -> None:
        rgbs = self.rgb_pending[camera]
        depths = self.depth_pending[camera]
        if not rgbs or not depths:
            return

        # Queues are tiny (normally <=4). Search all combinations so slightly
        # out-of-order DDS delivery still picks the closest valid timestamp.
        best: tuple[int, int, int] | None = None
        for rgb_i, (rgb_stamp, _) in enumerate(rgbs):
            for depth_i, (depth_stamp, _) in enumerate(depths):
                delta = abs(depth_stamp - rgb_stamp)
                if delta <= self.sync_slop_ns and (
                    best is None or delta < best[0]
                ):
                    best = (delta, rgb_i, depth_i)

        if best is None:
            self._drop_unmatchable_local_locked(camera)
            return

        _, rgb_i, depth_i = best
        rgb_stamp, rgb = rgbs.pop(rgb_i)
        _, depth = depths.pop(depth_i)
        pair = _RgbDepthPair(stamp_ns=rgb_stamp, rgb=rgb, depth=depth)

        pairs = self.local_pairs[camera]
        pairs.append(pair)
        pairs.sort(key=lambda item: item.stamp_ns)
        if len(pairs) > self.sync_queue_size:
            del pairs[: len(pairs) - self.sync_queue_size]

        # One callback can make more than one old pair resolvable after
        # out-of-order delivery. Drain all currently valid combinations.
        if rgbs and depths:
            self._pair_camera_locked(camera)

    def _drop_unmatchable_local_locked(self, camera: str) -> None:
        rgbs = self.rgb_pending[camera]
        depths = self.depth_pending[camera]
        if not rgbs or not depths:
            return

        newest_depth = depths[-1][0]
        while rgbs and rgbs[0][0] < newest_depth - self.sync_slop_ns:
            rgbs.pop(0)

        if not rgbs or not depths:
            return
        newest_rgb = rgbs[-1][0]
        while depths and depths[0][0] < newest_rgb - self.sync_slop_ns:
            depths.pop(0)

    def _select_multiview_candidate_locked(
        self,
    ) -> dict[str, _RgbDepthPair] | None:
        if any(camera not in self.camera_info for camera in self.camera_names):
            return None
        if any(not self.local_pairs[camera] for camera in self.camera_names):
            return None

        reference_camera = self.camera_names[0]
        reference_pairs = self.local_pairs[reference_camera]

        # Prefer the newest complete bundle because downstream processing is
        # explicitly latest-only. Other views are matched to camera_0 by their
        # camera-local RGB timestamps.
        for reference in reversed(reference_pairs):
            if (
                self.last_emitted_stamp_ns is not None
                and reference.stamp_ns <= self.last_emitted_stamp_ns
            ):
                continue

            candidate = {reference_camera: reference}
            complete = True
            for camera in self.camera_names[1:]:
                match = min(
                    self.local_pairs[camera],
                    key=lambda pair: abs(pair.stamp_ns - reference.stamp_ns),
                )
                if abs(match.stamp_ns - reference.stamp_ns) > self.multiview_slop_ns:
                    complete = False
                    break
                candidate[camera] = match

            if complete:
                return candidate

        self._drop_unmatchable_multiview_locked()
        return None

    def _drop_unmatchable_multiview_locked(self) -> None:
        if len(self.camera_names) < 2:
            return
        newest_oldest = min(
            pairs[-1].stamp_ns
            for pairs in self.local_pairs.values()
            if pairs
        )
        cutoff = newest_oldest - self.multiview_slop_ns
        for camera in self.camera_names:
            pairs = self.local_pairs[camera]
            while pairs and pairs[0].stamp_ns < cutoff:
                pairs.pop(0)

    def _candidate_is_current_locked(
        self,
        candidate: dict[str, _RgbDepthPair],
    ) -> bool:
        return all(
            any(pair is candidate[camera] for pair in self.local_pairs[camera])
            for camera in self.camera_names
        )

    def _consume_candidate_locked(
        self,
        candidate: dict[str, _RgbDepthPair],
    ) -> None:
        # Once a latest-only bundle is emitted, all older local pairs are stale.
        for camera in self.camera_names:
            chosen = candidate[camera]
            self.local_pairs[camera] = [
                pair
                for pair in self.local_pairs[camera]
                if pair.stamp_ns > chosen.stamp_ns
            ]

    def _world_from_camera(self, camera: str, stamp) -> np.ndarray | None:
        camera_frame = self.config.ros.camera_frame.format(camera=camera)
        timeout_s = max(0.0, float(self.config.ros.tf_timeout_ms)) / 1000.0
        try:
            transform = self.tf_buffer.lookup_transform(
                str(self.config.ros.world_frame),
                camera_frame,
                Time.from_msg(stamp),
                timeout=Duration(seconds=timeout_s),
            )
        except Exception as exc:
            now = time.monotonic()
            if now - self._last_tf_warning_s.get(camera, -1.0e9) >= 1.0:
                self._last_tf_warning_s[camera] = now
                self.node.get_logger().warning(
                    f"{camera}: missing TF {self.config.ros.world_frame} <- "
                    f"{camera_frame}: {exc}"
                )
            return None
        return _transform_matrix(transform)

    def _build_frame(
        self,
        candidate: dict[str, _RgbDepthPair],
    ) -> MultiCameraFrame | None:
        frames: dict[str, CameraFrameCpu] = {}
        for camera in self.camera_names:
            pair = candidate[camera]
            info = self.camera_info[camera]
            world_from_camera = self._world_from_camera(camera, pair.rgb.header.stamp)
            if world_from_camera is None:
                return None

            camera_frame = self.config.ros.camera_frame.format(camera=camera)
            frames[camera] = CameraFrameCpu(
                camera_name=camera,
                stamp_ns=pair.stamp_ns,
                rgb=_image_to_numpy(pair.rgb),
                depth=_depth_to_meters(pair.depth),
                K=np.asarray(info.k, dtype=np.float32).reshape(3, 3),
                T_world_camera=world_from_camera,
                optical_frame_id=camera_frame,
            )

        # camera_names[0] is the canonical multiview/temporal clock. Each view
        # retains its own RGB timestamp in CameraFrameCpu for upstream tracking.
        stamp_ns = candidate[self.camera_names[0]].stamp_ns
        return MultiCameraFrame(stamp_ns=stamp_ns, cameras=frames)
