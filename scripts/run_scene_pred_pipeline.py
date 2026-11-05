#!/usr/bin/env python3
"""ScenePredictor ROS node: multi-camera RGB-D in, tracked scene flow + scene cloud out (RViz topics).

Threading: scene_pred_pipeline/ros_loop.py (the umbrella PerceptiveSafetyFilter node uses the same loop)."""
from __future__ import annotations

import argparse
import sys

import rclpy
from rclpy.node import Node

from scene_pred_pipeline import ScenePredictionPipeline, load_config
from scene_pred_pipeline.ros_input import MultiCameraRosInput
from scene_pred_pipeline.ros_loop import GIL_SWITCH_INTERVAL_S, PerceptionLoop
from scene_pred_pipeline.ros_output import RosVisualizer


class ScenePredictorNode(Node):
    """Latest-only ROS input; one worker owns tracker + DifFlow state."""

    def __init__(self, config_path: str) -> None:
        super().__init__("scene_predictor")
        self.config = load_config(config_path)
        self.pipeline = ScenePredictionPipeline(self.config)
        self.visualizer = RosVisualizer(self, self.config, tracker_config=self.pipeline.tracker_config)
        self.loop = PerceptionLoop(
            self,
            self.pipeline,
            self.visualizer,
            profile_interval_frames=self.config.output.profile_interval_frames,
            max_frame_gap_s=self.config.flow.max_frame_gap_s,
        )
        self.input = MultiCameraRosInput(self, self.config, self.loop.enqueue)
        self.loop.start(self.input)
        self.get_logger().info("ScenePredictor ready: " + ", ".join(self.config.ros.camera_names))

    def destroy_node(self) -> bool:
        self.loop.close()
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/workspace/configs/default.yaml")
    args, ros_args = parser.parse_known_args()
    # Bound the GIL hand-over delay between the publisher and the GPU worker thread (CPython default 5 ms).
    sys.setswitchinterval(GIL_SWITCH_INTERVAL_S)
    rclpy.init(args=ros_args)
    node = ScenePredictorNode(args.config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
