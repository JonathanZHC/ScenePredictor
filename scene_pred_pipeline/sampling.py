from __future__ import annotations

import torch

from pointnet2 import pointnet2_utils

from .config import PipelineConfig
from .data_types import FlowInput, MotionState, TrackedFrame


def _even_candidate_indices(
    count: int,
    target: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= target:
        return torch.arange(count, device=device)
    return torch.linspace(0, count - 1, target, device=device).long()


def _pad_points(
    points: torch.Tensor,
    track_ids: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    count = points.shape[0]
    if count == 0:
        raise ValueError("Cannot pad an empty point cloud.")
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
        return points, track_ids
    indices = pointnet2_utils.furthest_point_sample(
        points[None].contiguous(),
        target,
    )[0].long()
    return points[indices], track_ids[indices]


class MovingPointSampler:
    """Step 5A: moving-point extraction, fast preselection and CUDA FPS."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def prepare(
        self,
        previous: TrackedFrame | None,
        current: TrackedFrame,
    ) -> FlowInput | None:
        if previous is None:
            return None

        previous_by_track = {
            item.track_id: item
            for item in previous.objects
        }
        selected = [
            item
            for item in current.objects
            if (
                item.motion_state == MotionState.MOVING
                and item.track_id in previous_by_track
            )
        ]
        if not selected:
            return None

        previous_points = torch.cat(
            [previous_by_track[item.track_id].points for item in selected],
            dim=0,
        )
        current_points = torch.cat(
            [item.points for item in selected],
            dim=0,
        )
        previous_ids = torch.cat(
            [
                torch.full(
                    (previous_by_track[item.track_id].points.shape[0],),
                    item.track_id,
                    device=item.points.device,
                    dtype=torch.int64,
                )
                for item in selected
            ]
        )
        current_ids = torch.cat(
            [
                torch.full(
                    (item.points.shape[0],),
                    item.track_id,
                    device=item.points.device,
                    dtype=torch.int64,
                )
                for item in selected
            ]
        )

        target = self.config.flow.target_points
        candidate_target = max(
            target,
            int(round(target * self.config.flow.pre_fps_factor)),
        )

        previous_candidate_indices = _even_candidate_indices(
            previous_points.shape[0],
            candidate_target,
            previous_points.device,
        )
        current_candidate_indices = _even_candidate_indices(
            current_points.shape[0],
            candidate_target,
            current_points.device,
        )
        previous_candidates = previous_points[previous_candidate_indices]
        current_candidates = current_points[current_candidate_indices]
        previous_candidate_ids = previous_ids[previous_candidate_indices]
        current_candidate_ids = current_ids[current_candidate_indices]

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
        dt_s = (current.stamp_ns - previous.stamp_ns) * 1.0e-9
        if dt_s <= 0.0:
            raise ValueError(f"Non-increasing frame timestamp: dt={dt_s}.")
        return FlowInput(
            previous_candidates=previous_candidates,
            current_candidates=current_candidates,
            previous_candidate_track_ids=previous_candidate_ids,
            current_candidate_track_ids=current_candidate_ids,
            previous_anchors=previous_anchors,
            current_anchors=current_anchors,
            previous_anchor_track_ids=previous_anchor_ids,
            current_anchor_track_ids=current_anchor_ids,
            dt_s=dt_s,
        )
