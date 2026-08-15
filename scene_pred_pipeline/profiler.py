from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
import time

import torch


class CycleProfiler:
    """Mixed CPU/CUDA stage profiler without a whole-device synchronization.

    CUDA stage events are recorded on ScenePredictor's current stream. At cycle
    end a fence event on that stream is synchronized, which makes elapsed-event
    timing valid without waiting for independent streams such as asynchronous
    SAM3 inference.
    """

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

    def finish(self) -> dict[str, float]:
        if self.cuda_enabled and self._cuda_events:
            # Wait only for ScenePredictor's current stream. A device-wide
            # torch.cuda.synchronize() would unnecessarily block SAM3's
            # independent asynchronous CUDA stream.
            fence = torch.cuda.Event()
            fence.record()
            fence.synchronize()

        timings: dict[str, float] = dict(self._recorded)
        timings.update(self._cpu_values)
        for name, (begin, end) in self._cuda_events.items():
            timings[name] = float(begin.elapsed_time(end))

        total_ms = 1000.0 * (time.perf_counter() - self._cpu_start)
        timings["cycle_total"] = total_ms
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

    def format_summary(self) -> str:
        rows = ["Cycle-time breakdown [ms]:"]
        for name, values in self.summary().items():
            rows.append(
                f"  {name:32s} mean={values['mean']:7.3f} "
                f"median={values['median']:7.3f} "
                f"p95={values['p95']:7.3f} "
                f"max={values['max']:7.3f}"
            )
        return "\n".join(rows)
