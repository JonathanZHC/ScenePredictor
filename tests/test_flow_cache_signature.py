from types import SimpleNamespace

import torch

from scene_pred_pipeline.data_types import FlowInput
from scene_pred_pipeline.flow_prediction import DifFlowPredictor


class _FakeRunner:
    def __init__(self):
        self.input_a = torch.empty((1, 2, 3), dtype=torch.float32)
        self.input_b = torch.empty((1, 2, 3), dtype=torch.float32)
        self._next_slot = 0
        self._previous_slot = None
        self._source = None
        self._target = None
        self.reset_count = 0

    @property
    def next_input(self):
        return self.input_a if self._next_slot == 0 else self.input_b

    def reset(self):
        self.reset_count += 1
        self._next_slot = 0
        self._previous_slot = None
        self._source = None
        self._target = None

    def replay_next(self):
        slot = self._next_slot
        current = self.input_a if slot == 0 else self.input_b
        if self._previous_slot is None:
            self._previous_slot = slot
            self._next_slot = 1 - slot
            return None
        previous = self.input_a if self._previous_slot == 0 else self.input_b
        self._source = previous
        self._target = current
        self._previous_slot = slot
        self._next_slot = 1 - slot
        return object()

    def flow(self):
        return torch.ones_like(self._source)

    def source_points(self):
        return self._source

    def warped_points(self):
        return self._source + 1.0


def _input(signature):
    previous = torch.zeros((2, 3), dtype=torch.float32)
    current = torch.ones((2, 3), dtype=torch.float32)
    ids = torch.tensor(signature, dtype=torch.int64)
    return FlowInput(
        previous_candidates=previous,
        current_candidates=current,
        previous_candidate_track_ids=ids,
        current_candidate_track_ids=ids,
        previous_anchors=previous,
        current_anchors=current,
        previous_anchor_track_ids=ids,
        current_anchor_track_ids=ids,
        current_dense_points=current,
        current_dense_track_ids=ids,
        common_track_ids=tuple(signature),
        dt_s=1.0,
    )


def _predictor():
    predictor = DifFlowPredictor.__new__(DifFlowPredictor)
    predictor.enabled = True
    predictor.runner = _FakeRunner()
    predictor.config = SimpleNamespace()
    predictor._cached_target_stamp_ns = None
    predictor._cached_track_signature = None
    return predictor


def test_streaming_cache_reuses_only_same_common_id_signature():
    predictor = _predictor()
    predictor.predict(_input((1, 2)), 10, 20)
    after_first = predictor.runner.reset_count

    # t=20 can be reused because the exact common-ID signature is unchanged.
    predictor.predict(_input((1, 2)), 20, 30)
    assert predictor.runner.reset_count == after_first

    # Same timestamp continuity but a different common-ID set must reset.
    predictor.predict(_input((2, 3)), 30, 40)
    assert predictor.runner.reset_count == after_first + 1
