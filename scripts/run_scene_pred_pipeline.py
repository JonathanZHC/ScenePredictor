#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import traceback

import rclpy
from rclpy.node import Node

from scene_pred_pipeline import (
    ScenePredictionPipeline,
    load_config,
)
from scene_pred_pipeline.ros_input import (
    MultiCameraRosInput,
)
from scene_pred_pipeline.ros_output import (
    RosVisualizer,
)


class ScenePredictorNode(Node):
    """Latest-only ROS input with one GPU-owning worker."""

    def __init__(
        self,
        config_path: str,
    ) -> None:
        super().__init__("scene_predictor")
        self.config = load_config(config_path)

        # Model loading, label validation and configured YOLOE dummy warmup
        # happen before subscriptions begin.
        self.pipeline = ScenePredictionPipeline(
            self.config
        )
        self.visualizer = RosVisualizer(
            self,
            self.config,
        )

        self._condition = threading.Condition()
        self._latest_frame = None
        self._stop_requested = False
        self.frame_count = 0

        self.input = MultiCameraRosInput(
            self,
            self.config,
            self._enqueue,
        )
        self._worker = threading.Thread(
            target=self._worker_loop,
            name="scene-predictor-gpu-worker",
            daemon=True,
        )
        self._worker.start()

        self.get_logger().info(
            "ScenePredictor ready.\n"
            f"  cameras: "
            f"{', '.join(self.config.ros.camera_names)}\n"
            f"  perception backend: "
            f"{self.config.models.backend}\n"
            f"  model: {self.config.models.weights}\n"
            f"  labels: "
            f"{self.config.models.label_file}\n"
            f"  profiling: "
            f"{self.config.runtime.enable_profiling}\n"
            f"  visualization: "
            f"{self.config.runtime.enable_visualization}"
        )

    def _enqueue(self, frame) -> None:
        # Latest-only queue prevents stale-frame latency.
        with self._condition:
            self._latest_frame = frame
            self._condition.notify()

    def _worker_loop(self) -> None:
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: (
                        self._latest_frame is not None
                        or self._stop_requested
                    )
                )
                if self._stop_requested:
                    return
                frame = self._latest_frame
                self._latest_frame = None

            try:
                output = self.pipeline.process(
                    frame
                )
                self.visualizer.publish(output)
                self.frame_count += 1

                gap_s = (
                    self.pipeline.last_flow_gap_s
                )
                if gap_s is not None:
                    self.get_logger().warning(
                        "Skipped one DifFlow3D pair "
                        f"because dt={gap_s:.6f}s "
                        "exceeded "
                        f"{self.config.flow.max_frame_gap_s:.6f}s."
                    )

                interval = (
                    self.config.output.profile_interval_frames
                )
                if (
                    self.config.runtime.enable_profiling
                    and interval > 0
                    and self.frame_count % interval == 0
                ):
                    self.get_logger().info(
                        "\n"
                        + self.pipeline.profiler.format_summary()
                    )
            except Exception:
                self.get_logger().error(
                    traceback.format_exc()
                )

    def destroy_node(self) -> bool:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join(timeout=5.0)
        return super().destroy_node()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the multi-camera ScenePredictor "
            "safety-filter pipeline."
        )
    )
    parser.add_argument(
        "--config",
        default="/workspace/configs/default.yaml",
        help="Pipeline YAML configuration.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    rclpy.init()
    node = ScenePredictorNode(args.config)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
