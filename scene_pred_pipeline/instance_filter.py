from __future__ import annotations

import torch

from .data_types import InstancePair, TrackedInstanceFrame


class CommonInstanceFilter:
    """Build exactly one t-1 -> t pair from persistent global track IDs.

    The common all-track case is zero-copy: both point and track-ID tensors are
    the packed CUDA buffers owned by their TrackedInstanceFrame.  A compacting
    gather is only performed when the common-ID intersection is a strict subset.
    """

    @staticmethod
    def _select_packed(
        frame: TrackedInstanceFrame,
        common_ids: tuple[int, ...],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if common_ids == frame.track_ids:
            return frame.packed_points_world, frame.packed_track_ids

        point_chunks: list[torch.Tensor] = []
        id_chunks: list[torch.Tensor] = []
        for track_id in common_ids:
            begin, end = frame.track_offsets[track_id]
            point_chunks.append(frame.packed_points_world[begin:end])
            id_chunks.append(frame.packed_track_ids[begin:end])

        if len(point_chunks) == 1:
            return point_chunks[0], id_chunks[0]
        return (
            torch.cat(point_chunks, dim=0).contiguous(),
            torch.cat(id_chunks, dim=0).contiguous(),
        )

    @staticmethod
    def select(
        previous: TrackedInstanceFrame | None,
        current: TrackedInstanceFrame,
    ) -> InstancePair | None:
        if previous is None:
            return None

        dt_s = (int(current.stamp_ns) - int(previous.stamp_ns)) * 1.0e-9
        if dt_s <= 0.0:
            # A stale/duplicate source frame is not a valid t-1 -> t flow pair.
            # The pipeline has already rebased its temporal state to ``current``,
            # so simply skip flow for this pair instead of terminating the
            # realtime worker.
            return None

        common_ids = tuple(sorted(set(previous.track_ids) & set(current.track_ids)))
        if not common_ids:
            return None

        previous_points, previous_ids = CommonInstanceFilter._select_packed(
            previous, common_ids
        )
        current_points, current_ids = CommonInstanceFilter._select_packed(
            current, common_ids
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
