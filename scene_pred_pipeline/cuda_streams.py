"""One high-priority CUDA stream shared by every latency-critical pipeline thread.

The tracker owner thread and the GPU worker thread both bind the *same* stream
object, so their mutual ordering is exactly what it was on the default stream
(no new cross-stream hazards), while the asynchronous SAM3 refresh keeps its
own default-priority (= lowest) stream. Whenever both have work queued, the
hardware scheduler dispatches the pipeline's blocks first, which stops the
~400 ms SAM3 job from doubling the latency of every GPU stage on the frames it
overlaps.
"""
from __future__ import annotations

import threading

import torch

_lock = threading.Lock()
_streams: dict[int, torch.cuda.Stream] = {}


def pipeline_stream(device: torch.device | str) -> torch.cuda.Stream | None:
    """Return the shared highest-priority stream for ``device`` (None on CPU)."""
    device = torch.device(device)
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    with _lock:
        stream = _streams.get(index)
        if stream is None:
            _, greatest = torch.cuda.Stream.priority_range()  # lower number = higher priority
            stream = torch.cuda.Stream(device=torch.device("cuda", index), priority=int(greatest))
            _streams[index] = stream
        return stream


def bind_pipeline_stream(device: torch.device | str) -> torch.cuda.Stream | None:
    """Make the shared high-priority stream current for the calling thread."""
    stream = pipeline_stream(device)
    if stream is not None:
        torch.cuda.set_stream(stream)
    return stream
