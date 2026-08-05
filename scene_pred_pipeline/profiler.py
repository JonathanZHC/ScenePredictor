from __future__ import annotations

from collections import defaultdict, deque
from contextlib import contextmanager
import time

import torch


class CycleProfiler:
    """Optional low-overhead cycle profiler.

    With profiling disabled every method is a no-op and finish() returns an
    empty dictionary. With CUDA timing enabled, one synchronization is performed
    at frame end and all GPU stages are resolved together.
    """

    def __init__(
        self,
        enabled: bool,
        use_cuda_events: bool = True,
        history_size: int = 300,
    ) -> None:
        self.enabled = bool(enabled)
        self.use_cuda_events = bool(
            self.enabled
            and use_cuda_events
            and torch.cuda.is_available()
        )
        self._cycle_start = 0.0
        self._events: dict[
            str,
            tuple[torch.cuda.Event, torch.cuda.Event],
        ] = {}
        self._cpu_starts: dict[str, float] = {}
        self._cpu_values: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    @contextmanager
    def stage(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def start_cycle(self) -> None:
        if not self.enabled:
            return
        self._cycle_start = time.perf_counter()
        self._events.clear()
        self._cpu_starts.clear()
        self._cpu_values.clear()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        if self.use_cuda_events:
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            self._events[name] = (begin, end)
        else:
            self._cpu_starts[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        if self.use_cuda_events:
            pair = self._events.get(name)
            if pair is None:
                raise KeyError(f"Profiler stage was not started: {name}")
            pair[1].record()
        else:
            begin = self._cpu_starts.pop(name, None)
            if begin is None:
                raise KeyError(f"Profiler stage was not started: {name}")
            self._cpu_values[name] = (
                1000.0 * (time.perf_counter() - begin)
            )

    def finish(self) -> dict[str, float]:
        if not self.enabled:
            return {}

        if self.use_cuda_events and self._events:
            torch.cuda.synchronize()

        timings: dict[str, float] = {}
        if self.use_cuda_events:
            for name, (begin, end) in self._events.items():
                value = float(begin.elapsed_time(end))
                timings[name] = value
                self._samples[name].append(value)
        else:
            for name, value in self._cpu_values.items():
                timings[name] = value
                self._samples[name].append(value)

        total_ms = 1000.0 * (
            time.perf_counter() - self._cycle_start
        )
        timings["cycle_total"] = total_ms
        self._samples["cycle_total"].append(total_ms)
        return timings

    def summary(self) -> dict[str, dict[str, float]]:
        if not self.enabled:
            return {}

        output: dict[str, dict[str, float]] = {}
        for name, samples in self._samples.items():
            if not samples:
                continue
            tensor = torch.tensor(
                list(samples),
                dtype=torch.float64,
            )
            output[name] = {
                "mean": float(tensor.mean()),
                "median": float(tensor.median()),
                "p95": float(torch.quantile(tensor, 0.95)),
                "max": float(tensor.max()),
                "count": float(tensor.numel()),
            }
        return output

    def format_summary(self) -> str:
        if not self.enabled:
            return "Cycle-time profiling disabled."

        rows = ["Cycle-time breakdown [ms]:"]
        for name, values in self.summary().items():
            rows.append(
                f"  {name:26s} "
                f"mean={values['mean']:7.3f} "
                f"median={values['median']:7.3f} "
                f"p95={values['p95']:7.3f} "
                f"max={values['max']:7.3f}"
            )
        return "\n".join(rows)
