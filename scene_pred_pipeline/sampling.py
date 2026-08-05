from __future__ import annotations

import torch
from pointnet2 import pointnet2_utils

from .config import PipelineConfig
from .data_types import (
    FlowCache,
    FlowInput,
    MotionState,
    TrackedFrame,
)


def _even_indices(
    count: int,
    target: int,
    device: torch.device,
) -> torch.Tensor:
    if count <= target:
        return torch.arange(
            count,
            device=device,
        )
    return torch.linspace(
        0,
        count - 1,
        target,
        device=device,
    ).long()


def _fps_once(
    points: torch.Tensor,
    track_ids: torch.Tensor,
    target: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """At most one CUDA FPS invocation for one frame."""

    if points.shape[0] == 0:
        raise ValueError(
            "Cannot sample an empty point cloud."
        )
    if points.shape[0] < target:
        indices = (
            torch.arange(
                target,
                device=points.device,
            )
            % points.shape[0]
        )
        return points[indices], track_ids[indices]

    indices = pointnet2_utils.furthest_point_sample(
        points[None].contiguous(),
        target,
    )[0].long()
    return points[indices], track_ids[indices]


class MovingPointSampler:
    """One FPS per current frame, cached for the next temporal pair."""

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config

    @staticmethod
    def _moving_points(
        frame: TrackedFrame,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        objects = [
            item
            for item in frame.objects
            if item.motion_state == MotionState.MOVING
        ]
        if not objects:
            return None

        points = torch.cat(
            [item.points for item in objects],
            dim=0,
        )
        track_ids = torch.cat(
            [
                torch.full(
                    (item.points.shape[0],),
                    item.track_id,
                    device=item.points.device,
                    dtype=torch.int64,
                )
                for item in objects
            ],
            dim=0,
        )
        return points, track_ids

    def cache_current(
        self,
        frame: TrackedFrame,
    ) -> FlowCache | None:
        """Compute and store the current frame's sampling result once."""

        if frame.flow_cache is not None:
            return frame.flow_cache

        moving = self._moving_points(frame)
        if moving is None:
            return None
        points, track_ids = moving

        target = self.config.flow.target_points
        candidate_target = max(
            target,
            int(
                round(
                    target
                    * self.config.flow.pre_fps_factor
                )
            ),
        )
        candidate_indices = _even_indices(
            points.shape[0],
            candidate_target,
            points.device,
        )
        candidates = points[candidate_indices]
        candidate_ids = track_ids[
            candidate_indices
        ]

        anchors, anchor_ids = _fps_once(
            candidates,
            candidate_ids,
            target,
        )
        frame.flow_cache = FlowCache(
            candidates=candidates,
            candidate_track_ids=candidate_ids,
            anchors=anchors,
            anchor_track_ids=anchor_ids,
        )
        return frame.flow_cache

    def prepare(
        self,
        previous: TrackedFrame | None,
        current: TrackedFrame,
    ) -> FlowInput | None:
        """Build a temporal pair from the exact stored frame caches.

        previous_cache.anchors is passed without filtering or resampling. It is
        therefore bit-identical to the current anchors used in the preceding
        call, which preserves DifFlow3DStreamingCudaGraphRunner's source-frame
        cache contract. If the moving-track set changes, the pair is skipped
        once and the pipeline emits conservative zero velocity for that
        transition instead of mixing unrelated scene-flow inputs.
        """

        if (
            previous is None
            or previous.flow_cache is None
            or current.flow_cache is None
        ):
            return None

        previous_cache = previous.flow_cache
        current_cache = current.flow_cache
        previous_tracks = torch.sort(
            torch.unique(
                previous_cache.candidate_track_ids
            )
        ).values
        current_tracks = torch.sort(
            torch.unique(
                current_cache.candidate_track_ids
            )
        ).values
        if (
            previous_tracks.shape != current_tracks.shape
            or not torch.equal(
                previous_tracks,
                current_tracks,
            )
        ):
            # Skip one flow pair whenever the moving-track set changes. This
            # avoids allowing a newly appearing/disappearing object to alter
            # the global scene-flow inference for established tracks. The
            # pipeline outputs current moving points with zero velocity for
            # this transition and resumes once the set is stable.
            return None

        dt_s = (
            current.stamp_ns - previous.stamp_ns
        ) * 1.0e-9
        if dt_s <= 0.0:
            raise ValueError(
                f"Non-increasing frame timestamp: "
                f"dt={dt_s}."
            )

        return FlowInput(
            previous_candidates=(
                previous_cache.candidates
            ),
            current_candidates=(
                current_cache.candidates
            ),
            previous_candidate_track_ids=(
                previous_cache.candidate_track_ids
            ),
            current_candidate_track_ids=(
                current_cache.candidate_track_ids
            ),
            previous_anchors=previous_cache.anchors,
            current_anchors=current_cache.anchors,
            previous_anchor_track_ids=(
                previous_cache.anchor_track_ids
            ),
            current_anchor_track_ids=(
                current_cache.anchor_track_ids
            ),
            dt_s=dt_s,
        )
