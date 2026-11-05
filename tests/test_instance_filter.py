import torch

from scene_pred_pipeline.data_types import TrackedInstance, TrackedInstanceFrame
from scene_pred_pipeline.instance_filter import CommonInstanceFilter


def _frame(stamp, ids):
    # Mirror TrackerAdapter: sorted IDs packed into one buffer, instances are views.
    ids = tuple(sorted(ids))
    packed_points = torch.tensor(
        [[float(track_id), 0.0, 0.0] for track_id in ids], dtype=torch.float32
    )
    packed_ids = torch.tensor(ids, dtype=torch.int32)
    offsets = {track_id: (i, i + 1) for i, track_id in enumerate(ids)}
    return TrackedInstanceFrame(
        frame_index=stamp,
        stamp_ns=stamp,
        instances=[
            TrackedInstance(
                global_track_id=track_id,
                semantic_label="object",
                points_world=packed_points[begin:end],
            )
            for track_id, (begin, end) in offsets.items()
        ],
        packed_points_world=packed_points,
        packed_track_ids=packed_ids,
        track_ids=ids,
        track_offsets=offsets,
    )


def test_common_ids_only():
    pair = CommonInstanceFilter.select(
        _frame(1_000_000_000, [1, 2, 4]),
        _frame(1_033_000_000, [1, 2, 3]),
    )
    assert pair is not None
    assert pair.common_track_ids == (1, 2)
    assert set(pair.previous_track_ids.tolist()) == {1, 2}
    assert set(pair.current_track_ids.tolist()) == {1, 2}


def test_no_common_ids():
    assert CommonInstanceFilter.select(
        _frame(1_000_000_000, [1]),
        _frame(1_033_000_000, [2]),
    ) is None
