"""One batched DifFlow3D forward whose rows are objects (object_flow.mode scheduled).

Every row holds the anchors of one object over a frame pair, in the row's own canonical frame (the source anchors'
centroid at the model centre, their RMS radius at the model radius, capped at ``k_max``), so every object gets its own
pyramid and nothing of another object. Rows share the forward's kernels: beside a large forward on another CUDA stream a
row of 256 anchors costs about 0.33 ms (512: 0.73 ms), against 1.8 ms for a forward of its own.

The forward (two encodes, decode, optional uncertainty head) is captured as a CUDA graph for a few row counts; a launch
uses the smallest captured count that fits and fills the unused rows with a copy of the last one.
"""
from __future__ import annotations

import torch


class RowBatch:
    def __init__(
        self,
        model,
        uncertainty_head,
        *,
        anchors: int,
        row_counts: tuple[int, ...],
        uncertainty: float,
        model_center: tuple[float, float, float],
        model_radius: float,
        k_max: float | None,
        device: torch.device,
        warmup: int = 3,
    ) -> None:
        self.model, self.head = model, uncertainty_head
        self.anchors = int(anchors)
        self.row_counts = tuple(sorted(int(count) for count in row_counts))
        self.capacity = self.row_counts[-1]
        self.uncertainty = float(uncertainty)
        self.center = torch.tensor(model_center, dtype=torch.float32, device=device)
        self.radius = float(model_radius)
        self.k_max = None if k_max is None else float(k_max)
        self.device = device
        self.warmup = int(warmup)
        self._graphs: dict[int, tuple] = {}
        self._stream: torch.cuda.Stream | None = None
        self._pending: tuple | None = None

    def prepare(self) -> None:
        """Capture the graphs (call where the pipeline captures its other DifFlow graphs)."""
        _, greatest = torch.cuda.Stream.priority_range()
        self._stream = torch.cuda.Stream(device=self.device, priority=int(greatest))
        for rows in self.row_counts:
            self._graphs[rows] = self._capture(rows)

    def _forward(self, source: torch.Tensor, target: torch.Tensor, scale: torch.Tensor):
        flow = self.model(source, target, source, target, None, uncertainty=self.uncertainty)[0][0][0]
        flow = flow.transpose(1, 2) if flow.shape[1] == 3 else flow
        cholesky = None
        if self.head is not None:
            features, certainty = self.model.recurrent0.head_inputs
            cholesky = self.head(source, features.transpose(1, 2), certainty.transpose(1, 2), flow, kk=scale)
        return flow, cholesky

    def _capture(self, rows: int) -> tuple:
        source = torch.randn(rows, self.anchors, 3, device=self.device) * 5.0 + self.center
        target = source + 0.05
        scale = torch.full((rows,), 10.0, device=self.device)
        unit = self.model.recurrent0
        keep_inputs = getattr(unit, "capture_head_inputs", False)
        unit.capture_head_inputs = self.head is not None
        try:
            with torch.inference_mode():
                for _ in range(self.warmup):
                    self._forward(source, target, scale)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    flow, cholesky = self._forward(source, target, scale)
                torch.cuda.synchronize(self.device)
        finally:
            unit.capture_head_inputs = keep_inputs
            unit.head_inputs = None
        return graph, source, target, scale, flow, cholesky

    def launch(self, source: torch.Tensor, target: torch.Tensor) -> None:
        """Start the forward for ``source`` / ``target`` ``[R, anchors, 3]`` (world, R <= capacity) on the batch's stream."""
        rows = int(source.shape[0])
        centroid = source.mean(1, keepdim=True)
        scale = self.radius / ((source - centroid) ** 2).sum(-1).mean(1).sqrt().clamp_min(1.0e-6)
        if self.k_max is not None:
            scale = scale.clamp(max=self.k_max)
        graph, graph_source, graph_target, graph_scale, _, _ = self._graphs[next(count for count in self.row_counts if count >= rows)]
        fill = torch.arange(graph_source.shape[0], device=self.device).clamp_(max=rows - 1)      # unused rows repeat the last one
        inputs = (
            (scale[:, None, None] * (source - centroid) + self.center)[fill],
            (scale[:, None, None] * (target - centroid) + self.center)[fill],
            scale[fill],
        )
        torch.cuda.current_stream(self.device).synchronize()          # the inputs are complete before the batch's stream reads them
        with torch.cuda.stream(self._stream):
            graph_source.copy_(inputs[0])
            graph_target.copy_(inputs[1])
            graph_scale.copy_(inputs[2])
            graph.replay()
        # The copies are only queued. ``inputs`` were allocated on the caller's stream: freed now, their memory could be
        # handed out again there before the batch's stream has read it, so they are kept until ``collect``.
        self._pending = (rows, scale, graph_source.shape[0], inputs)

    def collect(self) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Wait for the launched forward: displacement ``[R, anchors, 3]`` (m) and its covariance ``[R, anchors, 3, 3]`` (m^2)."""
        rows, scale, captured, _ = self._pending
        self._stream.synchronize()
        self._pending = None
        _, _, _, _, flow, cholesky = self._graphs[captured]
        displacement = flow[:rows] / scale[:, None, None]
        if cholesky is None:
            return displacement, None
        factor = cholesky[:rows].float()
        return displacement, (factor @ factor.transpose(-1, -2)) / (scale * scale)[:, None, None, None]
