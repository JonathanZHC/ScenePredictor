from __future__ import annotations

import torch

from .config import PipelineConfig
from .data_types import FlowInput, FlowResult


class VelocityRecovery:
    """Recover sparse source-anchor flow onto dense current tracked points."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _recover_track(
        self,
        target_points: torch.Tensor,
        warped_anchors: torch.Tensor,
        anchor_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if target_points.shape[0] == 0:
            return torch.empty_like(target_points)
        if warped_anchors.shape[0] == 0:
            return torch.zeros_like(target_points)

        k = min(int(self.config.recovery.knn), int(warped_anchors.shape[0]))
        temperature = max(float(self.config.recovery.temperature_m), 1.0e-6)
        chunk_size = max(1, int(self.config.recovery.chunk_size))
        outputs: list[torch.Tensor] = []

        for begin in range(0, target_points.shape[0], chunk_size):
            chunk = target_points[begin : begin + chunk_size]
            distances = torch.cdist(chunk, warped_anchors)
            values, indices = torch.topk(
                distances,
                k=k,
                dim=1,
                largest=False,
                sorted=False,
            )
            weights = torch.softmax(-values / temperature, dim=1)
            neighbors = anchor_velocity[indices]
            outputs.append(torch.sum(weights[..., None] * neighbors, dim=1))
        return torch.cat(outputs, dim=0)

    def recover(
        self,
        flow_input: FlowInput,
        flow_result: FlowResult,
    ) -> torch.Tensor:
        current = flow_input.current_dense_points
        if not self.config.recovery.restrict_same_track:
            return self._recover_track(
                current,
                flow_result.warped_anchors,
                flow_result.anchor_velocity,
            )

        output = torch.zeros_like(current)
        for track_id in flow_input.common_track_ids:
            target_mask = flow_input.current_dense_track_ids == int(track_id)
            source_mask = flow_input.previous_anchor_track_ids == int(track_id)
            if not bool(torch.any(target_mask)):
                continue
            output[target_mask] = self._recover_track(
                current[target_mask],
                flow_result.warped_anchors[source_mask],
                flow_result.anchor_velocity[source_mask],
            )
        return output
