from __future__ import annotations

import torch

from .config import PipelineConfig
from .data_types import FlowInput, FlowResult


class VelocityRecovery:
    """Recover source-anchor velocity to current moving candidates.

    Same-track restriction is applied as a GPU mask inside each cdist chunk.
    This avoids one Python loop and one GPU->CPU scalar transfer per track.
    Current-only tracks have no valid source anchors and receive zero velocity.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config

    def recover(
        self,
        flow_input: FlowInput,
        flow_result: FlowResult,
    ) -> torch.Tensor:
        current = flow_input.current_candidates
        anchors = flow_result.warped_anchors
        anchor_velocity = flow_result.anchor_velocity
        output = torch.zeros_like(current)

        if current.shape[0] == 0 or anchors.shape[0] == 0:
            return output

        k = min(
            self.config.recovery.knn,
            anchors.shape[0],
        )
        temperature = max(
            self.config.recovery.temperature_m,
            1.0e-8,
        )
        restrict = (
            self.config.recovery.restrict_same_track
        )
        anchor_ids = (
            flow_input.previous_anchor_track_ids
        )

        for begin in range(
            0,
            current.shape[0],
            self.config.recovery.chunk_size,
        ):
            end = min(
                begin
                + self.config.recovery.chunk_size,
                current.shape[0],
            )
            chunk = current[begin:end]
            distances = torch.cdist(
                chunk,
                anchors,
            )

            if restrict:
                target_ids = (
                    flow_input.current_candidate_track_ids[
                        begin:end
                    ]
                )
                same_track = (
                    target_ids[:, None]
                    == anchor_ids[None, :]
                )
                distances.masked_fill_(
                    ~same_track,
                    torch.inf,
                )

            values, indices = torch.topk(
                distances,
                k=k,
                dim=1,
                largest=False,
                sorted=False,
            )
            valid = torch.isfinite(values)
            logits = torch.where(
                valid,
                -values / temperature,
                torch.full_like(values, -1.0e9),
            )
            weights = torch.softmax(
                logits,
                dim=1,
            ) * valid
            weights = weights / weights.sum(
                dim=1,
                keepdim=True,
            ).clamp_min_(1.0e-12)
            neighbors = anchor_velocity[indices]
            output[begin:end] = torch.sum(
                weights[..., None] * neighbors,
                dim=1,
            )

        return output
