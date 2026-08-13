import torch

from scene_pred_pipeline.data_types import TrackedInstance, TrackedInstanceFrame
from scene_pred_pipeline.instance_filter import CommonInstanceFilter


def _frame(stamp, ids):
    return TrackedInstanceFrame(
        frame_index=stamp,
        stamp_ns=stamp,
        instances=[
            TrackedInstance(
                global_track_id=track_id,
                semantic_label="object",
                points_world=torch.tensor(
                    [[float(track_id), 0.0, 0.0]], dtype=torch.float32
                ),
            )
            for track_id in ids
        ],
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
