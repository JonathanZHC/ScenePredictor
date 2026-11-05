"""PerceptionLoop threading with stand-in node / pipeline / visualizer (no ROS, CPU only)."""
import threading
import time
from types import SimpleNamespace

import torch

from scene_pred_pipeline.ros_loop import PerceptionLoop


class _Logger:
    def __init__(self):
        self.errors, self.infos = [], []

    def info(self, text):
        self.infos.append(text)

    def warning(self, text):
        self.infos.append(text)

    def error(self, text):
        self.errors.append(text)


class _Pipeline:
    def __init__(self, fail_on=None):
        self.device = torch.device("cpu")
        self.last_flow_gap_s = None
        self.idle_marks = 0
        self.closed = False
        self.fail_on = fail_on
        self.profiler = SimpleNamespace(record_async=lambda *a: None, snapshot=lambda: {}, format_summary=lambda s: "summary")

    def process(self, frame):
        if frame == self.fail_on:
            raise RuntimeError("boom")
        return SimpleNamespace(frame=frame, flow_valid=False)

    def mark_gpu_idle(self):
        self.idle_marks += 1

    def close(self):
        self.closed = True


def _wait(predicate, timeout=5.0):
    end = time.monotonic() + timeout
    while not predicate() and time.monotonic() < end:
        time.sleep(0.005)
    return predicate()


def _loop(pipeline, after_process=None, interval=0):
    logger = _Logger()
    published = []
    visualizer = SimpleNamespace(publish=published.append, rate_summary=lambda: "rates")
    loop = PerceptionLoop(
        SimpleNamespace(get_logger=lambda: logger), pipeline, visualizer,
        profile_interval_frames=interval, max_frame_gap_s=0.2, after_process=after_process,
        extra_summary=lambda: "filter",
    )
    loop.start(SimpleNamespace(dropped_rgb_depth=0, dropped_multiview=0))
    return loop, published, logger


def test_frames_flow_through_worker_hook_and_publisher():
    seen = []
    pipeline = _Pipeline()
    loop, published, logger = _loop(pipeline, after_process=lambda out: seen.append(out.frame), interval=2)
    for frame in range(4):
        loop.enqueue(frame)
        assert _wait(lambda: loop.frame_count + loop.dropped_frames == frame + 1)
    assert _wait(lambda: len(published) + loop.publisher.dropped_outputs == loop.frame_count)
    loop.close()
    assert seen == [0, 1, 2, 3]          # one frame at a time: nothing dropped, the hook saw every frame
    assert pipeline.idle_marks == loop.frame_count and pipeline.closed
    assert any("summary" in t and "rates" in t and "filter" in t for t in logger.infos)
    assert not logger.errors


def test_exception_is_logged_and_gpu_marked_idle():
    pipeline = _Pipeline(fail_on=1)
    loop, published, logger = _loop(pipeline)
    loop.enqueue(1)
    assert _wait(lambda: pipeline.idle_marks == 1)
    loop.enqueue(2)
    assert _wait(lambda: len(published) == 1)
    loop.close()
    assert len(logger.errors) == 1 and "boom" in logger.errors[0]
    assert published[0].frame == 2


def test_latest_only_mailbox_drops_waiting_frames():
    gate = threading.Event()
    pipeline = _Pipeline()
    original = pipeline.process
    pipeline.process = lambda frame: (gate.wait(5.0), original(frame))[1]
    loop, published, _ = _loop(pipeline)
    loop.enqueue(0)                      # taken by the worker, which blocks on the gate
    assert _wait(lambda: loop._latest_frame is None)
    loop.enqueue(1)
    loop.enqueue(2)                      # replaces 1 while the worker is busy
    gate.set()
    assert _wait(lambda: loop.frame_count == 2)
    loop.close()
    assert loop.dropped_frames == 1
    assert [o.frame for o in published][-1] == 2
