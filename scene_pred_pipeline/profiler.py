from __future__ import annotations

from collections import defaultdict, deque
import time
from contextlib import contextmanager

import torch


class CycleProfiler:
    """GPU-event profiler with one synchronization at frame end."""

    def __init__(
        self,
        enabled: bool,
        history_size: int = 300,
    ) -> None:
        self.enabled = bool(enabled and torch.cuda.is_available())
        self._cpu_start = 0.0
        self._events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )

    @contextmanager
    def stage(self, name: str):
        """Profile one nested CUDA stage without synchronizing immediately."""
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def start_cycle(self) -> None:
        self._cpu_start = time.perf_counter()
        self._events.clear()

    def start(self, name: str) -> None:
        if not self.enabled:
            return
        begin = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        begin.record()
        self._events[name] = (begin, end)

    def stop(self, name: str) -> None:
        if not self.enabled:
            return
        event_pair = self._events.get(name)
        if event_pair is None:
            raise KeyError(f"Profiler stage was not started: {name}")
        event_pair[1].record()

    def finish(self) -> dict[str, float]:
        if self.enabled and self._events:
            torch.cuda.synchronize()

        timings: dict[str, float] = {}
        for name, (begin, end) in self._events.items():
            value = float(begin.elapsed_time(end))
            timings[name] = value
            self._samples[name].append(value)

        total_ms = 1000.0 * (time.perf_counter() - self._cpu_start)
        timings["cycle_total"] = total_ms
        self._samples["cycle_total"].append(total_ms)
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
                f"  {name:24s} mean={values['mean']:7.3f} "
                f"median={values['median']:7.3f} "
                f"p95={values['p95']:7.3f} "
                f"max={values['max']:7.3f}"
            )
        return "\n".join(rows)
