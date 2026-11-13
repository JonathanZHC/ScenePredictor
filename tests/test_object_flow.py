"""Per-object flow: context cells, slots, own-anchor merge (CPU, fake runners)."""
from pathlib import Path

import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig, RuntimeConfig, _load_difflow_config
from scene_pred_pipeline.data_types import FlowResult, InstancePair
from scene_pred_pipeline.object_flow import PerObjectFlowPredictor, context_cells, near_object

ROOT = Path(__file__).resolve().parents[1]


class _FakePredictor:
    """Returns the current points as anchors with their ids; remembers what it was given."""

    calibration_scale = None
    outlier_filter_enabled = False
    detailed_outlier_output = False

    def __init__(self):
        self.calls = []
        self.resets = 0
        self.reject = None                       # world position of a current point to reject
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None

    def prepare(self):
        pass

    def reset(self):
        self.resets += 1
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None

    def resolve_after_cycle_fence(self):
        return {"runner_decode": 2.0}, {"applied": False, "input_voxels": 3, "retained_voxels": 3, "removed_voxels": 0}, {}

    def predict(self, pair, *, track_labels=None):
        self.calls.append(pair)
        self._cached_target_stamp_ns = int(pair.current_stamp_ns)
        self._cached_track_signature = tuple(pair.common_track_ids)
        n = pair.current_points.shape[0]
        keep = torch.ones(n, dtype=torch.bool)
        if self.reject is not None:
            keep &= (pair.current_points - self.reject).norm(dim=1) > 1e-6
        return FlowResult(
            source_anchors=pair.current_points.clone(),
            warped_anchors=pair.current_points + 1.0,
            anchor_flow=torch.ones(n, 3),
            source_anchor_track_ids=pair.current_track_ids.clone(),
            target_input_keep_mask=keep,
            anchor_flow_cov=torch.eye(3).repeat(n, 1, 1),
            anchor_flow_cov_scale=torch.ones(n),
        )


def _predictor(max_objects=2, **per_object):
    config = PipelineConfig(runtime=RuntimeConfig(device="cpu"))
    if per_object:
        from dataclasses import replace
        config = replace(config, difflow=replace(config.difflow, per_object=replace(config.difflow.per_object, **per_object)))
    fakes = []

    def factory():
        fakes.append(_FakePredictor())
        return fakes[-1]

    predictor = PerObjectFlowPredictor(config, upstream_voxel_size_m=0.005, max_objects=max_objects, predictor_factory=factory)
    predictor.prepare()
    return predictor, fakes


def _blob(center, n, seed, spread=0.05):
    generator = torch.Generator().manual_seed(seed)
    return torch.tensor(center) + spread * torch.rand(n, 3, generator=generator)


def _pair(clouds, stamp=1, previous_stamp=0):
    """clouds: {track id: [N, 3]}; previous = current."""
    points = torch.cat(list(clouds.values()))
    ids = torch.cat([torch.full((cloud.shape[0],), track, dtype=torch.int32) for track, cloud in clouds.items()])
    return InstancePair(previous_stamp_ns=previous_stamp, current_stamp_ns=stamp, common_track_ids=tuple(clouds),
                        previous_points=points.clone(), current_points=points, previous_track_ids=ids.clone(),
                        current_track_ids=ids, dt_s=1 / 30)


def test_near_object_is_the_occupied_cells_and_their_neighbours():
    points = torch.tensor([[0.0, 0.0, 0.0], [0.10, 0.0, 0.0], [0.40, 0.0, 0.0], [2.0, 0.0, 0.0]])
    own = torch.tensor([True, False, False, False])
    near = near_object(context_cells(points, 0.15), own)
    assert near.tolist() == [True, True, False, False]


def test_far_objects_are_inferred_alone_and_all_anchors_come_back():
    predictor, fakes = _predictor()
    pair = _pair({1: _blob([0.0, 0.0, 1.0], 400, 0), 2: _blob([2.0, 0.0, 1.0], 300, 1)})
    result = predictor.predict(pair)
    sizes = sorted(fake.calls[0].current_points.shape[0] for fake in fakes)
    assert sizes == [300, 400]                                         # no context: nothing within a cell
    assert sorted(result.source_anchor_track_ids.tolist()) == sorted(pair.current_track_ids.tolist())
    assert result.target_input_keep_mask.all() and result.anchor_flow_cov.shape == (700, 3, 3)


def test_touching_objects_see_each_other_but_only_own_anchors_are_returned():
    predictor, fakes = _predictor()
    stick = torch.stack([torch.linspace(0.0, 0.6, 300), torch.zeros(300), torch.ones(300)], 1)        # along x from the hand
    person = _blob([-1.00, -0.50, 0.50], 600, 2, spread=1.00)                                          # reaches the stick's butt at x ~ 0
    result = predictor.predict(_pair({1: stick, 2: person}))
    stick_call = next(fake.calls[0] for fake in fakes if fake.calls[0].common_track_ids[0] == 1)
    context = stick_call.current_points[stick_call.current_track_ids == 2]
    assert 0 < context.shape[0] < 600                                  # the person's points near the stick, not all of them
    assert (stick_call.current_track_ids == 1).sum() == 300            # every own point
    assert result.source_anchors.shape[0] == 900                       # each point once: context estimates are dropped
    assert (result.source_anchor_track_ids == 1).sum() == 300 and (result.source_anchor_track_ids == 2).sum() == 600


def test_small_and_surplus_objects_ride_with_the_largest():
    predictor, fakes = _predictor(max_objects=2)
    pair = _pair({1: _blob([0.0, 0.0, 1.0], 600, 0), 2: _blob([2.0, 0.0, 1.0], 400, 1), 3: _blob([4.0, 0.0, 1.0], 300, 2),
                  4: _blob([6.0, 0.0, 1.0], 50, 3)})
    result = predictor.predict(pair)
    heads = {fake.calls[0].common_track_ids[0]: fake.calls[0] for fake in fakes}
    assert set(heads) == {1, 2}
    assert sorted(torch.unique(heads[1].current_track_ids).tolist()) == [1, 3, 4]      # 3 (no slot) and 4 (tiny) with the largest
    assert sorted(result.source_anchor_track_ids.tolist()) == sorted(pair.current_track_ids.tolist())


def test_one_object_is_the_joint_forward():
    predictor, fakes = _predictor()
    pair = _pair({7: _blob([0.0, 0.0, 1.0], 500, 0)})
    result = predictor.predict(pair)
    assert sum(len(fake.calls) for fake in fakes) == 1 and result.source_anchors.shape[0] == 500


def test_slots_are_kept_and_the_cached_source_is_not_rebuilt():
    predictor, fakes = _predictor()
    clouds = {1: _blob([0.0, 0.0, 1.0], 400, 0), 2: _blob([2.0, 0.0, 1.0], 300, 1)}
    predictor.predict(_pair(clouds, stamp=1, previous_stamp=0))
    first = {fake.calls[0].common_track_ids[0]: index for index, fake in enumerate(fakes)}
    predictor.predict(_pair(clouds, stamp=2, previous_stamp=1))
    assert {fake.calls[1].common_track_ids[0]: index for index, fake in enumerate(fakes)} == first
    assert all(fake.calls[1].previous_points.shape[0] == 700 for fake in fakes)        # cached: the pair's cloud is passed through unread
    predictor.predict(_pair({2: clouds[2]}, stamp=3, previous_stamp=2))               # object 1 left: object 2 stays in its slot
    assert len(fakes[first[2]].calls) == 3 and len(fakes[first[1]].calls) == 2


def test_rejected_dense_points_map_back_to_the_full_cloud():
    predictor, fakes = _predictor()
    pair = _pair({1: _blob([0.0, 0.0, 1.0], 400, 0), 2: _blob([2.0, 0.0, 1.0], 300, 1)})
    for fake in fakes:
        fake.reject = pair.current_points[450].clone()                 # a point of object 2
    result = predictor.predict(pair)
    assert (~result.target_input_keep_mask).nonzero().flatten().tolist() == [450]


def test_reset_and_resolve_cover_every_slot():
    predictor, fakes = _predictor()
    predictor.predict(_pair({1: _blob([0.0, 0.0, 1.0], 400, 0), 2: _blob([2.0, 0.0, 1.0], 300, 1)}))
    timing, info, _ = predictor.resolve_after_cycle_fence()
    assert timing == {"runner_decode": 2.0, "runner_objects": 2.0} and info["input_voxels"] == 6
    predictor.reset()
    assert all(fake.resets == 1 for fake in fakes)


def test_config_section(tmp_path):
    text = (ROOT / "configs" / "difflow.yaml").read_text()
    config = _load_difflow_config(ROOT / "configs" / "difflow.yaml")
    assert config.per_object.enabled and config.per_object.context_cell_m == 0.15 and config.per_object.max_objects == 4
    path = tmp_path / "difflow.yaml"
    path.write_text(text.replace("context_cell_m: 0.15", "context_cell_m: 0.2").replace("enabled: true\n  max_objects", "enabled: false\n  max_objects"))
    config = _load_difflow_config(path)
    assert not config.per_object.enabled and config.per_object.context_cell_m == 0.2
    for bad in ("context_cell_m: 0.0", "context_cell_m: 0.15\n  cell: 1"):
        path.write_text(text.replace("context_cell_m: 0.15", bad))
        with pytest.raises(ValueError):
            _load_difflow_config(path)
    path.write_text(text.replace("max_objects: 4", "max_objects: 0"))
    with pytest.raises(ValueError):
        _load_difflow_config(path)
