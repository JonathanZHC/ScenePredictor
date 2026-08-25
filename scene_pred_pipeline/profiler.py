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

    _TRACKER = (
        "tracking_model",
        "postprocess",
        "alignment",
        "adapter",
        "tracker_other",
    )
    _DIFFLOW_RUNNER = (
        ("runner_voxel2", "voxel-2"),
        ("runner_outlier_filter", "outlier filter"),
        ("runner_final_selection", "final selection"),
        ("runner_stage_scale", "stage + scale"),
        ("runner_encode", "encode"),
        ("runner_decode", "decode"),
    )
    _DIFFLOW_DETAIL = _DIFFLOW_RUNNER + (("difflow_other", "other"),)
    _ASYNC = (
        "sam3_async",
        "sam3_filter",
        "sam3_slot_assoc",
        # ROS message building runs on the publisher thread, off the numerical
        # critical path; reported here so its cost stays visible.
        "publish_total",
    )
    _POST_CYCLE_DEBUG = ("outlier_debug",)

    def __init__(
        self,
        enabled: bool,
        history_size: int = 300,
    ) -> None:
        self.cuda_enabled = bool(enabled and torch.cuda.is_available())
        self._cpu_start = 0.0
        # Event pairs are created once per stage name and reused every cycle
        # (cudaEventCreate/Destroy per stage per frame is avoidable overhead).
        self._event_pool: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._cuda_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
        self._cpu_starts: dict[str, float] = {}
        self._cpu_values: dict[str, float] = {}
        self._recorded: dict[str, float] = {}
        self._samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._outlier_samples: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=history_size)
        )
        self._outlier_config: dict[str, object] = {}

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
            pair = self._event_pool.get(name)
            if pair is None:
                pair = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                self._event_pool[name] = pair
            pair[0].record()
            self._cuda_events[name] = pair
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

    def record_async(self, name: str, value_ms: float) -> None:
        """Append a sample measured on another thread (not part of cycle_total).

        ``deque.append`` is atomic under the GIL, so this is safe to call from
        the publisher thread while the worker records its own stages.
        """
        self._samples[str(name)].append(float(value_ms))

    def snapshot(self) -> dict[str, object]:
        """Cheap copy of the sample history for off-thread summary formatting."""
        return {
            "samples": {name: list(values) for name, values in self._samples.items()},
            "outlier_samples": {
                name: list(values) for name, values in self._outlier_samples.items()
            },
            "outlier_config": dict(self._outlier_config),
        }

    def record_difflow_timings(
        self,
        values_ms: dict[str, float],
        total_ms: float,
    ) -> float:
        """Record leaf runner timings and return unprofiled DifFlow overhead."""
        for name, value in values_ms.items():
            self._samples[str(name)].append(float(value))
        other_ms = self._residual(
            total_ms,
            *(values_ms.get(name, 0.0) for name, _ in self._DIFFLOW_RUNNER),
        )
        self._samples["difflow_other"].append(other_ms)
        return other_ms

    def record_outlier_filter(self, info: dict[str, object]) -> None:
        """Accumulate non-timing filter diagnostics for periodic summaries."""
        self._outlier_config = {
            key: info[key]
            for key in (
                "enabled",
                "min_component_size_ratio",
            )
            if key in info
        }
        self._outlier_samples["applied_ratio"].append(
            float(bool(info.get("applied", False)))
        )
        for source, target in (
            ("input_voxels", "input_voxels"),
            ("retained_voxels", "retained_voxels"),
            ("dense_input_points", "dense_input_points"),
            ("dense_removed_points", "dense_removed_points"),
        ):
            self._outlier_samples[target].append(float(info.get(source, 0)))

        statistics = info.get("statistics")
        if not isinstance(statistics, dict):
            return
        for name, value in statistics.items():
            if value is not None:
                self._outlier_samples[str(name)].append(float(value))

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
        ):
            timings.setdefault(name, 0.0)

        timings["tracker_other"] = self._residual(
            timings["tracker_total"],
            timings["tracking_model"],
            timings["postprocess"],
            timings["alignment"],
            timings["adapter"],
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

    @staticmethod
    def _summarize(
        sample_groups: dict[str, deque[float]],
        *,
        include_sum: bool = False,
    ) -> dict[str, dict[str, float]]:
        output: dict[str, dict[str, float]] = {}
        for name, samples in sample_groups.items():
            if not samples:
                continue
            tensor = torch.tensor(list(samples), dtype=torch.float64)
            values = {
                "mean": float(tensor.mean()),
                "median": float(tensor.median()),
                "p95": float(torch.quantile(tensor, 0.95)),
                "max": float(tensor.max()),
                "count": float(tensor.numel()),
            }
            if include_sum:
                values["sum"] = float(tensor.sum())
            output[name] = values
        return output

    def summary(self, snapshot: dict[str, object] | None = None) -> dict[str, dict[str, float]]:
        samples = self._samples if snapshot is None else snapshot["samples"]
        return self._summarize(samples)

    def outlier_summary(self, snapshot: dict[str, object] | None = None) -> dict[str, dict[str, float]]:
        samples = self._outlier_samples if snapshot is None else snapshot["outlier_samples"]
        return self._summarize(samples, include_sum=True)

    @staticmethod
    def _format_row(label: str, values: dict[str, float], *, indent: int = 2) -> str:
        return (
            f"{' ' * indent}{label:28s} "
            f"mean={values['mean']:7.3f} "
            f"median={values['median']:7.3f} "
            f"p95={values['p95']:7.3f} "
            f"max={values['max']:7.3f}"
        )

    def format_summary(self, snapshot: dict[str, object] | None = None) -> str:
        """Format the rolling summary; pass ``snapshot()`` to run off-thread."""
        values = self.summary(snapshot)
        outlier_config = (
            self._outlier_config if snapshot is None else snapshot["outlier_config"]
        )
        rows = ["End-to-end numerical cycle [ms]:"]

        def add(name: str, indent: int = 2, label: str | None = None) -> None:
            item = values.get(name)
            if item is not None:
                rows.append(self._format_row(label or name, item, indent=indent))

        add("cycle_total")
        rows.append("")
        add("tracker_total")
        for name in self._TRACKER:
            add(name, 4)
        rows.append("")
        add("instance_filter")
        rows.append("")
        add("difflow_total")
        for name, label in self._DIFFLOW_DETAIL:
            add(name, 4, label)
        rows.append("")
        add("velocity_recovery")
        add("cycle_other")

        async_present = any(name in values for name in self._ASYNC)
        if async_present:
            rows.extend(("", "Async diagnostics [not part of cycle_total]:"))
            for name in self._ASYNC:
                add(name)

        debug_present = any(name in values for name in self._POST_CYCLE_DEBUG)
        if debug_present:
            rows.extend(("", "Post-cycle debug diagnostics [not part of cycle_total]:"))
            for name in self._POST_CYCLE_DEBUG:
                add(name)

        outlier = self.outlier_summary(snapshot)
        if bool(outlier_config.get("enabled", False)) and outlier:
            rows.extend(("", "Voxel outlier filter diagnostics:"))
            rows.append(
                "  config: "
                "min_component_size_ratio="
                f"{float(outlier_config.get('min_component_size_ratio', 0.05)):.4f}"
            )

            def add_outlier(name: str, label: str, *, show_sum: bool = False) -> None:
                item = outlier.get(name)
                if item is None:
                    return
                suffix = f" total={item['sum']:.0f}" if show_sum else ""
                rows.append(
                    f"  {label:32s} "
                    f"mean={item['mean']:8.3f} "
                    f"median={item['median']:8.3f} "
                    f"p95={item['p95']:8.3f} "
                    f"max={item['max']:8.3f}{suffix}"
                )

            add_outlier("applied_ratio", "filter applied ratio")
            add_outlier("input_voxels", "voxel-2 input")
            add_outlier("retained_voxels", "voxels retained")
            add_outlier("dense_input_points", "dense input points")
            add_outlier(
                "dense_removed_points",
                "dense points removed",
                show_sum=True,
            )
            add_outlier("component_count", "blocks in scene")
            add_outlier("instance_count", "instances in scene")
            add_outlier("largest_component_voxels", "largest block [voxels]")
            add_outlier(
                "instances_with_removed_components",
                "instances with removals",
                show_sum=True,
            )
            add_outlier("removed_component_count", "removed blocks", show_sum=True)
            add_outlier("removed_voxel_count", "removed voxels", show_sum=True)
        return "\n".join(rows)
