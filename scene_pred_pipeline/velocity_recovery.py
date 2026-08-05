from __future__ import annotations

import torch

from .config import PipelineConfig
from .data_types import FlowInput, FlowResult


class VelocityRecovery:
    """Step 5C: recover source-anchor velocity to every current moving point."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _recover_track(
        self,
        target_points: torch.Tensor,
        warped_anchors: torch.Tensor,
        anchor_velocity: torch.Tensor,
    ) -> torch.Tensor:
        if warped_anchors.shape[0] == 0:
            return torch.zeros_like(target_points)
        k = min(self.config.recovery.knn, warped_anchors.shape[0])
        outputs: list[torch.Tensor] = []
        temperature = self.config.recovery.temperature_m

        for begin in range(
            0,
            target_points.shape[0],
            self.config.recovery.chunk_size,
        ):
            chunk = target_points[
                begin : begin + self.config.recovery.chunk_size
            ]
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
            outputs.append(
                torch.sum(weights[..., None] * neighbors, dim=1)
            )
        return torch.cat(outputs, dim=0)

    def recover(
        self,
        flow_input: FlowInput,
        flow_result: FlowResult,
    ) -> torch.Tensor:
        current = flow_input.current_candidates
        output = torch.zeros_like(current)

        if not self.config.recovery.restrict_same_track:
            return self._recover_track(
                current,
                flow_result.warped_anchors,
                flow_result.anchor_velocity,
            )

        for track_id in torch.unique(
            flow_input.current_candidate_track_ids
        ).tolist():
            target_mask = (
                flow_input.current_candidate_track_ids == track_id
            )
            source_mask = (
                flow_input.previous_anchor_track_ids == track_id
            )
            output[target_mask] = self._recover_track(
                current[target_mask],
                flow_result.warped_anchors[source_mask],
                flow_result.anchor_velocity[source_mask],
            )
        return output
