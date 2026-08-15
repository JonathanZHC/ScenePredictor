from __future__ import annotations

import torch

from .data_types import InstancePair, TrackedInstanceFrame


class CommonInstanceFilter:
    """Build exactly one t-1 -> t pair from persistent global track IDs."""

    @staticmethod
    def select(
        previous: TrackedInstanceFrame | None,
        current: TrackedInstanceFrame,
    ) -> InstancePair | None:
        if previous is None:
            return None

        dt_s = (int(current.stamp_ns) - int(previous.stamp_ns)) * 1.0e-9
        if dt_s <= 0.0:
            raise ValueError(f"Non-increasing frame timestamp: dt={dt_s:.9f}s")

        previous_by_id = {
            int(item.global_track_id): item
            for item in previous.instances
            if item.points_world.numel() > 0
        }
        current_by_id = {
            int(item.global_track_id): item
            for item in current.instances
            if item.points_world.numel() > 0
        }
        common_ids = tuple(sorted(previous_by_id.keys() & current_by_id.keys()))
        if not common_ids:
            return None

        previous_chunks = [
            previous_by_id[track_id].points_world for track_id in common_ids
        ]
        current_chunks = [
            current_by_id[track_id].points_world for track_id in common_ids
        ]
        previous_points = torch.cat(previous_chunks, dim=0).contiguous()
        current_points = torch.cat(current_chunks, dim=0).contiguous()

        device = current_points.device
        previous_ids = torch.cat(
            [
                torch.full(
                    (points.shape[0],),
                    track_id,
                    device=device,
                    dtype=torch.int32,
                )
                for track_id, points in zip(common_ids, previous_chunks)
            ],
            dim=0,
        )
        current_ids = torch.cat(
            [
                torch.full(
                    (points.shape[0],),
                    track_id,
                    device=device,
                    dtype=torch.int32,
                )
                for track_id, points in zip(common_ids, current_chunks)
            ],
            dim=0,
        )

        return InstancePair(
            previous_stamp_ns=int(previous.stamp_ns),
            current_stamp_ns=int(current.stamp_ns),
            common_track_ids=common_ids,
            previous_points=previous_points,
            current_points=current_points,
            previous_track_ids=previous_ids,
            current_track_ids=current_ids,
            dt_s=dt_s,
        )
