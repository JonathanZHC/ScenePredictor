from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
import time

import torch


class CycleProfiler:
    """Low-overhead end-to-end ScenePredictor profiler.

    CUDA events are recorded on ScenePredictor's current stream and resolved by
    one fence event at cycle end. Tracker-internal timings are imported as
    already-resolved measurements from MultiViewRGBDTracker, so this profiler
    never adds a second synchronization to the tracker/SAM3 streams.
    """

    _TOP_LEVEL = (
        "cycle_total",
        "tracker_total",
        "instance_filter",
        "difflow_total",
        "velocity_recovery",
        "cycle_other",
    )
    _TRACKER = (
        "tracking_model",
        "postprocess",
        "alignment",
        "adapter",
        "tracker_other",
    )
    _DIFFLOW = (
        "source_prepare",
        "target_prepare",
        "inference",
        "output_extract",
        "difflow_other",
    )
    _ASYNC = (
        "sam3_async",
        "sam3_filter",
        "sam3_slot_assoc",
    )

    def __init__(
        self,
        enabled: bool,
        history_size: int = 300,
    ) -> None:
        self.cuda_enabled = bool(enabled and torch.cuda.is_available())
        self._cpu_start = 0.0
        self._cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._cpu_starts: dict[str, float] = {}
        self._cpu_values: dict[str, float] = {}
        self._recorded: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    @contextmanager
    def stage(self, name: str, *, cuda: bool = True):
        self.start(name, cuda=cuda)
        try:
            yield
        finally:
            self.stop(name)

    def start_cycle(self) -> None:
        self._cpu_start = time.perf_counter()
        self._cuda_events.clear()
        self._cpu_starts.clear()
        self._cpu_values.clear()
        self._recorded.clear()

    def start(self, name: str, *, cuda: bool = True) -> None:
        if cuda and self.cuda_enabled:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            self._cuda_events[name] = (begin, end)
        else:
            self._cpu_starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if name in self._cuda_events:
            self._cuda_events[name][1].record()
            return
        started = self._cpu_starts.pop(name, None)
        if started is None:
            raise KeyError(f"Profiler stage was not started: {name}")
        self._cpu_values[name] = 1000.0 * (time.perf_counter() - started)

    def record(self, name: str, value_ms: float) -> None:
        self._recorded[str(name)] = float(value_ms)

    @staticmethod
    def _residual(total: float, *parts: float) -> float:
        # Do not clamp. A materially negative residual is useful evidence that a
        # supposedly additive timing boundary has accidentally become nested.
        return float(total) - sum(float(value) for value in parts)

    def finish(self) -> dict[str, float]:
        if self.cuda_enabled and self._cuda_events:
            # Wait only for ScenePredictor's current stream. A device-wide
            # synchronize would unnecessarily block asynchronous SAM3 work.
            fence = torch.cuda.Event()
            fence.record()
            fence.synchronize()

        timings: dict[str, float] = dict(self._recorded)
        timings.update(self._cpu_values)
        for name, (begin, end) in self._cuda_events.items():
            timings[name] = float(begin.elapsed_time(end))

        timings["cycle_total"] = 1000.0 * (time.perf_counter() - self._cpu_start)

        # Make every numerical decomposition additive on each individual frame.
        for name in (
            "tracker_total",
            "instance_filter",
            "difflow_total",
            "velocity_recovery",
            "tracking_model",
            "postprocess",
            "alignment",
            "adapter",
            "source_prepare",
            "target_prepare",
            "inference",
            "output_extract",
        ):
            timings.setdefault(name, 0.0)

        timings["tracker_other"] = self._residual(
            timings["tracker_total"],
            timings["tracking_model"],
            timings["postprocess"],
            timings["alignment"],
            timings["adapter"],
        )
        timings["difflow_other"] = self._residual(
            timings["difflow_total"],
            timings["source_prepare"],
            timings["target_prepare"],
            timings["inference"],
            timings["output_extract"],
        )
        timings["cycle_other"] = self._residual(
            timings["cycle_total"],
            timings["tracker_total"],
            timings["instance_filter"],
            timings["difflow_total"],
            timings["velocity_recovery"],
        )

        for name, value in timings.items():
            self._samples[name].append(float(value))
        return timings

    def summary(self) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for name, samples in self._samples.items():
            if not samples:
                continue
            tensor = torch.tensor(list(samples), dtype=torch.float64)
            output[name] = {
                "mean": float(tensor.mean()),
                "median": float(tensor.median()),
                "p95": float(torch.quantile(tensor, 0.95)),
                "max": float(tensor.max()),
                "count": float(tensor.numel()),
            }
        return output

    @staticmethod
    def _format_row(name: str, values: dict[str, float], *, indent: int = 2) -> str:
        return (
            f"{' ' * indent}{name:28s} "
            f"mean={values['mean']:7.3f} "
            f"median={values['median']:7.3f} "
            f"p95={values['p95']:7.3f} "
            f"max={values['max']:7.3f}"
        )

    def format_summary(self) -> str:
        values = self.summary()
        rows = ["End-to-end numerical cycle [ms]:"]

        def add(name: str, indent: int = 2) -> None:
            item = values.get(name)
            if item is not None:
                rows.append(self._format_row(name, item, indent=indent))

        add("cycle_total")
        rows.append("")
        add("tracker_total")
        for name in self._TRACKER:
            add(name, 4)
        rows.append("")
        add("instance_filter")
        rows.append("")
        add("difflow_total")
        for name in self._DIFFLOW:
            add(name, 4)
        rows.append("")
        add("velocity_recovery")
        add("cycle_other")

        async_present = any(name in values for name in self._ASYNC)
        if async_present:
            rows.extend(("", "Async diagnostics [not part of cycle_total]:"))
            for name in self._ASYNC:
                add(name)
        return "\n".join(rows)
