"""Threading of a ROS node around ScenePredictionPipeline (ScenePredictor's node and the umbrella safety-filter node).

    RGB-D bundle --enqueue--> [one-slot mailbox] --> GPU worker thread: pipeline.process(frame), after_process(output)
                                                                     --> [one-slot mailbox] --> publisher thread: RViz

Both mailboxes are latest-only: a frame that arrives while the worker is busy replaces the waiting one (dropped_frames),
an output the publisher has not taken yet is replaced by the next (dropped_outputs). Nothing queues.
"""
from __future__ import annotations

import gc
import threading
import time
import traceback
from typing import Any, Callable

import torch

from .cuda_streams import bind_pipeline_stream

# Periodic explicit garbage collection once the automatic collector is disabled (see _configure_gc).
_GC_INTERVAL_S = 5.0
# The worker thread drops the GIL in every torch/warp call and must win it back from the publisher thread
# (marker/cloud/image building in Python). CPython hands the GIL over only every switch interval (5 ms
# default), so a busy publisher can delay each re-acquisition by up to 5 ms; nodes set this instead.
GIL_SWITCH_INTERVAL_S = 5.0e-4


def _maybe_collect(last_gc: float) -> bool:
    if not gc.isenabled() and time.monotonic() - last_gc >= _GC_INTERVAL_S:
        gc.collect()
        return True
    return False


def _configure_gc() -> None:
    """Freeze the (large, static) startup heap and stop automatic collection.

    Every frame allocates a few hundred short-lived objects; with the automatic collector on, gen-2 passes
    periodically traverse the whole model/tensor object graph and stall the worker for tens of ms. After warm-up
    the heap is frozen out of the collector's view and the publisher thread collects every few seconds instead.
    """
    gc.collect()
    gc.freeze()
    gc.disable()


class _PublishWorker:
    """Latest-only visualization/logging thread: ROS message building (PointCloud2 packing, markers, RGB overlays)
    and the periodic profiler summary run here, so the GPU worker returns to the next frame immediately."""

    def __init__(self, loop: PerceptionLoop, name: str) -> None:
        self.loop = loop
        self._condition = threading.Condition()
        self._item: tuple | None = None
        self._stop = False
        self.dropped_outputs = 0
        device = loop.pipeline.device
        self._stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._thread = threading.Thread(target=self._loop, name=name, daemon=True)

    def start(self) -> None:
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
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def _loop(self) -> None:
        torch.set_grad_enabled(False)          # grad mode is per thread; nothing here needs autograd
        if self._stream is not None:
            torch.cuda.set_stream(self._stream)
        loop = self.loop
        last_gc = time.monotonic()
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._item is not None or self._stop, timeout=1.0)
                if self._stop:
                    return
                item, self._item = self._item, None
            if item is not None:
                output, ready_event, summary_snapshot = item
                try:
                    if ready_event is not None and self._stream is not None:
                        self._stream.wait_event(ready_event)     # the output was produced on the pipeline stream
                    started = time.perf_counter()
                    loop.visualizer.publish(output)
                    if self._stream is not None:
                        # Finish this stream's reads of the output before its tensors are released: the caching
                        # allocator would otherwise hand their memory to the worker stream while reads are pending.
                        self._stream.synchronize()
                    loop.pipeline.profiler.record_async("publish_total", 1000.0 * (time.perf_counter() - started))
                    if summary_snapshot is not None:
                        loop.log_summary(summary_snapshot)
                except Exception:
                    loop.logger.error(traceback.format_exc())
            if _maybe_collect(last_gc):
                last_gc = time.monotonic()


class PerceptionLoop:
    """The GPU worker + publisher threads of a node. Usage:

        loop = PerceptionLoop(node, pipeline, visualizer, profile_interval_frames=..., max_frame_gap_s=...)
        ros_input = MultiCameraRosInput(node, config, loop.enqueue)
        loop.start(ros_input)
        ...
        loop.close()          # joins both threads, closes the pipeline

    after_process(output) runs on the worker thread right after pipeline.process (the safety filter of the umbrella
    node); the GPU is marked idle for a cooperative SAM3 refresh after it at the latest. extra_summary() is appended
    to the periodic profiler summary.
    """

    def __init__(
        self,
        node: Any,
        pipeline: Any,
        visualizer: Any,
        *,
        profile_interval_frames: int,
        max_frame_gap_s: float,
        after_process: Callable[[Any], None] | None = None,
        extra_summary: Callable[[], str] | None = None,
        thread_prefix: str = "scene-predictor",
    ) -> None:
        self.node = node
        self.logger = node.get_logger()
        self.pipeline = pipeline
        self.visualizer = visualizer
        self.input: Any = None
        self._interval = int(profile_interval_frames)
        self._max_frame_gap_s = float(max_frame_gap_s)
        self._after_process = after_process
        self._extra_summary = extra_summary
        self._condition = threading.Condition()
        self._latest_frame = None
        self._stop_requested = False
        self.frame_count = 0
        self.dropped_frames = 0
        self.publisher = _PublishWorker(self, f"{thread_prefix}-publisher")
        self._worker = threading.Thread(target=self._worker_loop, name=f"{thread_prefix}-gpu-worker", daemon=True)

    def start(self, ros_input: Any) -> None:
        self.input = ros_input
        self.publisher.start()
        self._worker.start()

    def enqueue(self, frame) -> None:
        with self._condition:
            if self._latest_frame is not None:
                self.dropped_frames += 1
            self._latest_frame = frame
            self._condition.notify()

    def log_summary(self, snapshot) -> None:
        text = self.pipeline.profiler.format_summary(snapshot)
        text += (
            f"\n  input: dropped_bundles={self.dropped_frames}"
            f" unmatched_rgbd={self.input.dropped_rgb_depth}"
            f" unmatched_multiview={self.input.dropped_multiview}"
            f" dropped_outputs={self.publisher.dropped_outputs}"
        )
        text += "\n  " + self.visualizer.rate_summary()
        if self._extra_summary is not None:
            text += "\n" + self._extra_summary()
        self.logger.info("\n" + text)

    def _worker_loop(self) -> None:
        # All CUDA work of this thread goes on the shared high-priority pipeline stream (the tracker's stream).
        stream = bind_pipeline_stream(self.pipeline.device)
        torch.set_grad_enabled(False)          # grad mode is per thread; nothing here needs autograd
        gc_configured = False
        while True:
            with self._condition:
                self._condition.wait_for(lambda: self._latest_frame is not None or self._stop_requested)
                if self._stop_requested:
                    return
                frame, self._latest_frame = self._latest_frame, None
            try:
                output = self.pipeline.process(frame)
                self.frame_count += 1
                if self.frame_count == 1:
                    self.logger.info("first perception cycle completed (models warm)")
                if not gc_configured and self.frame_count >= 10:
                    _configure_gc()
                    gc_configured = True
                gap_s = self.pipeline.last_flow_gap_s
                if gap_s is not None:
                    self.logger.warning(
                        f"Skipped one DifFlow3D pair and rebased to the current tracker frame "
                        f"(dt={gap_s:.6f}s > max_frame_gap_s {self._max_frame_gap_s:.6f}s)."
                    )
                if self._after_process is not None:
                    self._after_process(output)

                # Anchor tensors alias DifFlow's CUDA-graph buffers, which the next replay overwrites, so they are
                # copied before the hand-over (tiny); everything else is frame-owned.
                if output.flow_valid:
                    output.source_anchors = output.source_anchors.clone()
                    output.warped_anchors = output.warped_anchors.clone()
                ready_event = None
                if stream is not None:
                    ready_event = torch.cuda.Event()
                    ready_event.record(stream)
                snapshot = None
                if self._interval > 0 and self.frame_count % self._interval == 0:
                    snapshot = self.pipeline.profiler.snapshot()
                self.publisher.submit(output, ready_event, snapshot)
            except Exception:
                self.logger.error(traceback.format_exc())
            finally:
                # Normally already done (by process(), or by after_process with pipeline.defer_gpu_idle); here for
                # exceptions, so a cooperative SAM3 refresh is never left paused.
                self.pipeline.mark_gpu_idle()

    def close(self) -> None:
        with self._condition:
            self._stop_requested = True
            self._condition.notify_all()
        if self._worker.is_alive():
            self._worker.join()
        self.publisher.stop()
        try:
            self.pipeline.close()
        except Exception:
            self.logger.error("Pipeline shutdown failed:\n" + traceback.format_exc())
