from types import SimpleNamespace

import torch

from scene_pred_pipeline.data_types import InstancePair
from scene_pred_pipeline.flow_prediction import DifFlowPredictor


class _FakeRunner:
    """Minimal streaming runner: buffers one staged frame, pairs it with the next."""

    def __init__(self):
        self.reset_count = 0
        self._previous = None
        self._staged = None
        self._source = None
        self._source_ids = None
        self._target = None

    def reset(self):
        self.reset_count += 1
        self._previous = None
        self._staged = None

    def begin_profile_window(self):
        pass

    def stage_world(self, points, point_ids=None):
        self._staged = (points, point_ids)

    def replay_next(self):
        current = self._staged
        previous = self._previous
        self._previous = current
        if previous is None:
            return None
        self._source, self._source_ids = previous
        self._target = current[0]
        return object()

    def flow_world(self):
        return torch.ones_like(self._source)[None]

    def source_points_world(self):
        return self._source[None]

    def warped_points_world(self):
        return (self._source + 1.0)[None]

    def source_point_ids(self):
        return self._source_ids

    def target_input_keep_mask(self):
        return torch.ones(self._target.shape[0], dtype=torch.bool)

    def target_preprocess_info(self):
        return {}


def _pair(signature, previous_stamp_ns, current_stamp_ns):
    previous = torch.zeros((2, 3), dtype=torch.float32)
    current = torch.ones((2, 3), dtype=torch.float32)
    ids = torch.tensor(signature, dtype=torch.int64)
    return InstancePair(
        previous_stamp_ns=previous_stamp_ns,
        current_stamp_ns=current_stamp_ns,
        common_track_ids=tuple(signature),
        previous_points=previous,
        current_points=current,
        previous_track_ids=ids,
        current_track_ids=ids,
        dt_s=1.0,
    )


def _predictor():
    predictor = DifFlowPredictor.__new__(DifFlowPredictor)
    predictor.runner = _FakeRunner()
    predictor.uncertainty_head = None
    predictor.calibration_scale = None
    predictor._cached_target_stamp_ns = None
    predictor._cached_track_signature = None
    predictor._last_target_preprocess_info = None
    predictor._debug_pending_current = None
    # Detailed outlier debug output disabled, so no debug candidates are captured.
    predictor.config = SimpleNamespace(
        difflow=SimpleNamespace(
            preprocessing=SimpleNamespace(
                outlier_filter=SimpleNamespace(enabled=False, detailed_output=False)
            )
        )
    )
    return predictor


def test_streaming_cache_reuses_only_same_common_id_signature():
    predictor = _predictor()
    predictor.predict(_pair((1, 2), 10, 20))
    after_first = predictor.runner.reset_count

    # t=20 can be reused because the exact common-ID signature is unchanged.
    predictor.predict(_pair((1, 2), 20, 30))
    assert predictor.runner.reset_count == after_first

    # Same timestamp continuity but a different common-ID set must reset.
    predictor.predict(_pair((2, 3), 30, 40))
    assert predictor.runner.reset_count == after_first + 1
