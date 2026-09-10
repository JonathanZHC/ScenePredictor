#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import threading
import time
import traceback

import rclpy
import torch
from rclpy.node import Node

from scene_pred_pipeline import ScenePredictionPipeline, load_config
from scene_pred_pipeline.cuda_streams import bind_pipeline_stream
from scene_pred_pipeline.ros_input import MultiCameraRosInput
from scene_pred_pipeline.ros_output import RosVisualizer

# Periodic explicit garbage collection interval once the automatic collector is
# disabled (see ScenePredictorNode._configure_gc).
_GC_INTERVAL_S = 5.0


class _PublishWorker:
    """Latest-only visualization/logging thread.

    ROS message building (PointCloud2 packing, markers, RGB overlays) and the
    periodic profiler summary run here so the numerical worker returns to the
    next frame immediately. The mailbox holds one item: if publishing falls
    behind, older outputs are dropped, never queued.
    """

    def __init__(self, node: ScenePredictorNode) -> None:
        self.node = node
        self._condition = threading.Condition()
        self._item: tuple | None = None
        self._stop = False
        self.dropped_outputs = 0
        self._stream = (
            torch.cuda.Stream(device=node.pipeline.device)
            if node.pipeline.device.type == "cuda"
            else None
        )
        self._thread = threading.Thread(
            target=self._loop, name="scene-predictor-publisher", daemon=True
        )
        self._thread.start()

    def submit(self, output, ready_event, summary_snapshot) -> None:
        with self._condition:
            if self._item is not None:
                self.dropped_outputs += 1
            self._item = (output, ready_event, summary_snapshot)
            self._condition.notify()

    def stop(self) -> None:
        with self._condition:
            self._stop = True
            self._condition.notify_all()
        self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        if self._stream is not None:
            torch.cuda.set_stream(self._stream)
        last_gc = time.monotonic()
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._item is not None or self._stop, timeout=1.0
                )
                if self._stop:
                    return
                item, self._item = self._item, None
            if item is None:
                if self._maybe_collect(last_gc):
                    last_gc = time.monotonic()
                continue
            output, ready_event, summary_snapshot = item
            try:
                if ready_event is not None and self._stream is not None:
                    # Output tensors were produced on the pipeline stream.
                    self._stream.wait_event(ready_event)
                started = time.perf_counter()
                self.node.visualizer.publish(output)
                publish_ms = 1000.0 * (time.perf_counter() - started)
                self.node.pipeline.profiler.record_async("publish_total", publish_ms)
                if summary_snapshot is not None:
                    text = self.node.pipeline.profiler.format_summary(summary_snapshot)
                    text += (
                        f"\n  input: dropped_bundles={self.node.dropped_frames}"
                        f" unmatched_rgbd={self.node.input.dropped_rgb_depth}"
                        f" unmatched_multiview={self.node.input.dropped_multiview}"
                        f" dropped_outputs={self.dropped_outputs}"
                    )
                    self.node.get_logger().info("\n" + text)
            except Exception:
                self.node.get_logger().error(traceback.format_exc())
            if self._maybe_collect(last_gc):
                last_gc = time.monotonic()

    @staticmethod
    def _maybe_collect(last_gc: float) -> bool:
        if not gc.isenabled() and time.monotonic() - last_gc >= _GC_INTERVAL_S:
            gc.collect()
            return True
        return False


class ScenePredictorNode(Node):
    """Latest-only ROS input; one worker owns tracker + DifFlow state."""

    def __init__(self, config_path: str) -> None:
        super().__init__("scene_predictor")
        self.config = load_config(config_path)
        self.pipeline = ScenePredictionPipeline(self.config)
        self.visualizer = RosVisualizer(
            self,
            self.config,
            tracker_config=self.pipeline.tracker_config,
        )

        self._condition = threading.Condition()
        self._latest_frame = None
        self._stop_requested = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.publisher = _PublishWorker(self)
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
            "ScenePredictor ready: " + ", ".join(self.config.ros.camera_names)
        )

    @staticmethod
    def _configure_gc() -> None:
        """Freeze the (large, static) startup heap and stop automatic collection.

        Every frame allocates a few hundred short-lived objects; with the
        automatic collector on, gen-2 passes periodically traverse the whole
        model/tensor object graph and stall the worker for tens of ms. After
        warm-up the heap is frozen out of the collector's view and an explicit
        collect runs on the publisher thread every few seconds instead.
        """
        gc.collect()
        gc.freeze()
        gc.disable()

    def _enqueue(self, frame) -> None:
        with self._condition:
            if self._latest_frame is not None:
                self.dropped_frames += 1
            self._latest_frame = frame
            self._condition.notify()

    def _worker_loop(self) -> None:
        # All DifFlow/recovery CUDA work of this thread goes on the shared
        # high-priority pipeline stream (same stream object the tracker owner uses).
        stream = bind_pipeline_stream(self.pipeline.device)
        gc_configured = False
        interval = int(self.config.output.profile_interval_frames)
        while True:
            with self._condition:
                self._condition.wait_for(
                    lambda: self._latest_frame is not None or self._stop_requested
                )
                if self._stop_requested:
                    return
                frame = self._latest_frame
                self._latest_frame = None

            try:
                output = self.pipeline.process(frame)
                self.frame_count += 1

                if not gc_configured and self.frame_count >= 10:
                    self._configure_gc()
                    gc_configured = True

                gap_s = self.pipeline.last_flow_gap_s
                if gap_s is not None:
                    self.get_logger().warning(
                        "Skipped one DifFlow3D pair and rebased to the current "
                        f"tracker frame because dt={gap_s:.6f}s exceeded "
                        f"{self.config.flow.max_frame_gap_s:.6f}s."
                    )

                # Hand the frame to the publisher thread. Anchor tensors alias
                # DifFlow's CUDA-graph buffers, which the next replay overwrites,
                # so they are copied here (tiny); everything else is frame-owned.
                if output.flow_valid:
                    output.source_anchors = output.source_anchors.clone()
                    output.warped_anchors = output.warped_anchors.clone()
                ready_event = None
                if stream is not None:
                    ready_event = torch.cuda.Event()
                    ready_event.record(stream)
                snapshot = None
                if interval > 0 and self.frame_count % interval == 0:
                    snapshot = self.pipeline.profiler.snapshot()
                self.publisher.submit(output, ready_event, snapshot)
            except Exception:
                self.get_logger().error(traceback.format_exc())

    def destroy_node(self) -> bool:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        self._worker.join()
        self.publisher.stop()
        try:
            self.pipeline.close()
        except Exception:
            self.get_logger().error("Pipeline shutdown failed:\n" + traceback.format_exc())
        return super().destroy_node()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="/workspace/configs/default.yaml")
    args, ros_args = parser.parse_known_args()
    rclpy.init(args=ros_args)
    node = ScenePredictorNode(args.config)
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
