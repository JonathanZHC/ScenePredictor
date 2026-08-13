from __future__ import annotations

import torch
from pointnet2 import pointnet2_utils

from .config import PipelineConfig
from .data_types import FlowInput, InstancePair


def _even_candidate_indices(
    count: int,
    target: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= target:
        return torch.arange(count, device=device, dtype=torch.long)
    return torch.linspace(0, count - 1, target, device=device).long()


def _pad_points(
    points: torch.Tensor,
    track_ids: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = int(points.shape[0])
    if count == 0:
        raise ValueError("Cannot pad an empty point cloud")
    if count >= target:
        return points, track_ids
    repeated = torch.arange(target, device=points.device) % count
    return points[repeated], track_ids[repeated]


def _fps(
    points: torch.Tensor,
    track_ids: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    points, track_ids = _pad_points(points, track_ids, target)
    if points.shape[0] == target:
        return points.contiguous(), track_ids.contiguous()
    indices = pointnet2_utils.furthest_point_sample(
        points[None].contiguous(),
        target,
    )[0].long()
    return points[indices].contiguous(), track_ids[indices].contiguous()


def _repair_coverage(
    full_points: torch.Tensor,
    full_ids: torch.Tensor,
    selected_points: torch.Tensor,
    selected_ids: torch.Tensor,
    common_track_ids: tuple[int, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Guarantee one selected point per common ID without evicting another ID.

    ``selected_points`` is large (4096 candidates / 2048 anchors in the default
    configuration) while the instance count is small, so an overrepresented donor
    always exists whenever the selected capacity can represent every common ID.
    """

    if len(common_track_ids) > int(selected_points.shape[0]):
        return selected_points, selected_ids

    counts = {
        int(track_id): int(torch.sum(selected_ids == int(track_id)).item())
        for track_id in common_track_ids
    }
    for track_id in common_track_ids:
        track_id = int(track_id)
        if counts[track_id] > 0:
            continue

        full_index = torch.nonzero(full_ids == track_id, as_tuple=False)
        if full_index.numel() == 0:
            continue

        donor_ids = [
            donor_id for donor_id, count in counts.items() if count > 1
        ]
        if not donor_ids:
            break
        donor_mask = torch.zeros_like(selected_ids, dtype=torch.bool)
        for donor_id in donor_ids:
            donor_mask |= selected_ids == int(donor_id)
        donor_positions = torch.nonzero(donor_mask, as_tuple=False)
        if donor_positions.numel() == 0:
            break
        # Replace from the tail so the leading deterministic FPS order stays intact.
        donor_position = int(donor_positions[-1, 0].item())
        donor_id = int(selected_ids[donor_position].item())
        source = int(full_index[0, 0])
        selected_points[donor_position].copy_(full_points[source])
        selected_ids[donor_position] = track_id
        counts[donor_id] -= 1
        counts[track_id] = 1

    return selected_points, selected_ids


class TrackedPointSampler:
    """Candidate reduction + one global CUDA FPS for each side of the pair."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def prepare(self, pair: InstancePair) -> FlowInput:
        target = int(self.config.flow.target_points)
        if target < 1024:
            raise ValueError("DifFlow3D optimized runner requires target_points >= 1024")

        candidate_target = max(
            target,
            int(round(target * float(self.config.flow.pre_fps_factor))),
        )

        previous_indices = _even_candidate_indices(
            pair.previous_points.shape[0],
            candidate_target,
            pair.previous_points.device,
        )
        current_indices = _even_candidate_indices(
            pair.current_points.shape[0],
            candidate_target,
            pair.current_points.device,
        )

        previous_candidates = pair.previous_points[previous_indices].contiguous()
        current_candidates = pair.current_points[current_indices].contiguous()
        previous_candidate_ids = pair.previous_track_ids[previous_indices].contiguous()
        current_candidate_ids = pair.current_track_ids[current_indices].contiguous()

        previous_candidates, previous_candidate_ids = _repair_coverage(
            pair.previous_points,
            pair.previous_track_ids,
            previous_candidates,
            previous_candidate_ids,
            pair.common_track_ids,
        )
        current_candidates, current_candidate_ids = _repair_coverage(
            pair.current_points,
            pair.current_track_ids,
            current_candidates,
            current_candidate_ids,
            pair.common_track_ids,
        )

        previous_anchors, previous_anchor_ids = _fps(
            previous_candidates,
            previous_candidate_ids,
            target,
        )
        current_anchors, current_anchor_ids = _fps(
            current_candidates,
            current_candidate_ids,
            target,
        )

        previous_anchors, previous_anchor_ids = _repair_coverage(
            previous_candidates,
            previous_candidate_ids,
            previous_anchors,
            previous_anchor_ids,
            pair.common_track_ids,
        )
        current_anchors, current_anchor_ids = _repair_coverage(
            current_candidates,
            current_candidate_ids,
            current_anchors,
            current_anchor_ids,
            pair.common_track_ids,
        )

        return FlowInput(
            previous_candidates=previous_candidates,
            current_candidates=current_candidates,
            previous_candidate_track_ids=previous_candidate_ids,
            current_candidate_track_ids=current_candidate_ids,
            previous_anchors=previous_anchors,
            current_anchors=current_anchors,
            previous_anchor_track_ids=previous_anchor_ids,
            current_anchor_track_ids=current_anchor_ids,
            current_dense_points=pair.current_points,
            current_dense_track_ids=pair.current_track_ids,
            common_track_ids=pair.common_track_ids,
            dt_s=float(pair.dt_s),
        )
