"""object_flow: mode selection, context grid, per_object slots and own-anchor merge (CPU, fake runners)."""
from pathlib import Path

import pytest
import torch

import scene_pred_pipeline.object_flow as object_flow
from flow_fakes import FakePredictor, blob, config, pair
from scene_pred_pipeline.config import _load_difflow_config
from scene_pred_pipeline.object_flow import ObjectLayout, PerObjectFlowPredictor, SlotMap, build_flow_predictor
from scene_pred_pipeline.scheduled_flow import ScheduledFlowPredictor

ROOT = Path(__file__).resolve().parents[1]


def _predictor(max_objects=2, **settings):
    fakes = []

    def factory():
        fakes.append(FakePredictor())
        return fakes[-1]

    predictor = PerObjectFlowPredictor(config(**settings), upstream_voxel_size_m=0.005, max_objects=max_objects, predictor_factory=factory)
    predictor.prepare()
    return predictor, fakes


def _layout(clouds, cell_m=0.15):
    frame = pair(clouds)
    return ObjectLayout.of(frame.current_points, frame.current_track_ids, torch.tensor(sorted(clouds), dtype=torch.int32), cell_m)


# ---- mode selection --------------------------------------------------------------------------------------------------
def test_the_mode_selects_the_flow_stage(monkeypatch):
    class Joint:
        def __init__(self, config, *, upstream_voxel_size_m):
            pass

    monkeypatch.setattr(object_flow, "DifFlowPredictor", Joint)
    build = lambda **settings: build_flow_predictor(config(**settings), upstream_voxel_size_m=0.005)
    assert type(build()) is Joint                                                  # the default
    per_object = build(mode="per_object", max_objects=3)
    assert type(per_object) is PerObjectFlowPredictor and len(per_object.pool) == 3
    scheduled = build(mode="scheduled", scheduled={"own_forwards": 3})
    assert type(scheduled) is ScheduledFlowPredictor and len(scheduled.pool) == 3


def test_a_single_trackable_instance_is_always_the_joint_forward(monkeypatch):
    class Joint:
        def __init__(self, config, *, upstream_voxel_size_m):
            pass

    monkeypatch.setattr(object_flow, "DifFlowPredictor", Joint)
    for mode in ("per_object", "scheduled"):
        assert type(build_flow_predictor(config(instances=1, mode=mode), upstream_voxel_size_m=0.005)) is Joint


# ---- context grid ----------------------------------------------------------------------------------------------------
def test_an_objects_cloud_is_its_cells_and_their_neighbours():
    points = {1: torch.tensor([[0.0, 0.0, 0.0]]), 2: torch.tensor([[0.10, 0.0, 0.0], [0.40, 0.0, 0.0], [2.0, 0.0, 0.0]])}
    layout = _layout(points)
    assert layout.cloud_mask(torch.tensor([0])).tolist() == [True, True, False, False]
    assert layout.count.tolist() == [1.0, 3.0] and layout.extent.tolist() == pytest.approx([0.0, 1.9])


def test_touching_is_decided_on_the_grid():
    layout = _layout({1: blob([0.0, 0.0, 1.0], 50, 0), 2: blob([0.12, 0.0, 1.0], 50, 1), 3: blob([2.0, 0.0, 1.0], 50, 2)})
    touching = layout.touching()
    assert touching[0, 1] and touching[1, 0] and not touching[0, 2] and not touching[2, 1]


def test_evenly_spaced_points_are_the_objects_own_in_cloud_order():
    clouds = {5: blob([0.0, 0.0, 1.0], 40, 0), 9: blob([2.0, 0.0, 1.0], 100, 1)}
    layout = _layout(clouds)
    sample = layout.evenly_spaced([1, 0], 100)
    assert sample.shape == (2, 100, 3) and torch.equal(sample[0], clouds[9])
    assert torch.equal(sample[1, 0], clouds[5][0]) and torch.equal(sample[1, -1], clouds[5][-1])     # 40 points, repeated


def test_a_key_keeps_its_slot():
    slots = SlotMap(3)
    assert slots.assign([7, 8]) == {7: 0, 8: 1}
    assert slots.assign([8, 9]) == {8: 1, 9: 0}


# ---- per_object ------------------------------------------------------------------------------------------------------
def test_far_objects_are_inferred_alone_and_all_anchors_come_back():
    predictor, fakes = _predictor()
    frame = pair({1: blob([0.0, 0.0, 1.0], 400, 0), 2: blob([2.0, 0.0, 1.0], 300, 1)})
    result = predictor.predict(frame)
    sizes = sorted(fake.calls[0].current_points.shape[0] for fake in fakes)
    assert sizes == [300, 400]                                         # no context: nothing within a cell
    assert sorted(result.source_anchor_track_ids.tolist()) == sorted(frame.current_track_ids.tolist())
    assert result.target_input_keep_mask.all() and result.anchor_flow_cov.shape == (700, 3, 3)


def test_touching_objects_see_each_other_but_only_own_anchors_are_returned():
    predictor, fakes = _predictor()
    stick = torch.stack([torch.linspace(0.0, 0.6, 300), torch.zeros(300), torch.ones(300)], 1)        # along x from the hand
    person = blob([-1.00, -0.50, 0.50], 600, 2, spread=1.00)                                           # reaches the stick's butt at x ~ 0
    result = predictor.predict(pair({1: stick, 2: person}))
    stick_call = next(fake.calls[0] for fake in fakes if fake.calls[0].common_track_ids[0] == 1)
    context = stick_call.current_points[stick_call.current_track_ids == 2]
    assert 0 < context.shape[0] < 600                                  # the person's points near the stick, not all of them
    assert (stick_call.current_track_ids == 1).sum() == 300            # every own point
    assert result.source_anchors.shape[0] == 900                       # each point once: context estimates are dropped
    assert (result.source_anchor_track_ids == 1).sum() == 300 and (result.source_anchor_track_ids == 2).sum() == 600


def test_small_and_surplus_objects_ride_with_the_largest():
    predictor, fakes = _predictor(max_objects=2)
    frame = pair({1: blob([0.0, 0.0, 1.0], 600, 0), 2: blob([2.0, 0.0, 1.0], 400, 1), 3: blob([4.0, 0.0, 1.0], 300, 2),
                  4: blob([5.5, 0.0, 1.0], 50, 3)})
    result = predictor.predict(frame)
    heads = {fake.calls[0].common_track_ids[0]: fake.calls[0] for fake in fakes}
    assert set(heads) == {1, 2}
    assert sorted(torch.unique(heads[1].current_track_ids).tolist()) == [1, 3, 4]      # 3 (no slot) and 4 (tiny) with the largest
    assert sorted(result.source_anchor_track_ids.tolist()) == sorted(frame.current_track_ids.tolist())


def test_one_object_is_the_joint_forward():
    predictor, fakes = _predictor()
    result = predictor.predict(pair({7: blob([0.0, 0.0, 1.0], 500, 0)}))
    assert sum(len(fake.calls) for fake in fakes) == 1 and result.source_anchors.shape[0] == 500


def test_slots_are_kept_and_the_cached_source_is_not_rebuilt():
    predictor, fakes = _predictor()
    clouds = {1: blob([0.0, 0.0, 1.0], 400, 0), 2: blob([2.0, 0.0, 1.0], 300, 1)}
    predictor.predict(pair(clouds, stamp=1, previous_stamp=0))
    first = {fake.calls[0].common_track_ids[0]: index for index, fake in enumerate(fakes)}
    predictor.predict(pair(clouds, stamp=2, previous_stamp=1))
    assert {fake.calls[1].common_track_ids[0]: index for index, fake in enumerate(fakes)} == first
    assert all(fake.calls[1].previous_points.shape[0] == 700 for fake in fakes)        # cached: the pair's cloud is passed through unread
    predictor.predict(pair({2: clouds[2]}, stamp=3, previous_stamp=2))                # object 1 left: object 2 stays in its slot
    assert len(fakes[first[2]].calls) == 3 and len(fakes[first[1]].calls) == 2


def test_rejected_dense_points_map_back_to_the_full_cloud():
    predictor, fakes = _predictor()
    frame = pair({1: blob([0.0, 0.0, 1.0], 400, 0), 2: blob([2.0, 0.0, 1.0], 300, 1)})
    for fake in fakes:
        fake.reject = frame.current_points[450].clone()                # a point of object 2
    result = predictor.predict(frame)
    assert (~result.target_input_keep_mask).nonzero().flatten().tolist() == [450]


def test_reset_and_resolve_cover_every_slot():
    predictor, fakes = _predictor()
    predictor.predict(pair({1: blob([0.0, 0.0, 1.0], 400, 0), 2: blob([2.0, 0.0, 1.0], 300, 1)}))
    timing, info, _ = predictor.resolve_after_cycle_fence()
    assert timing == {"runner_decode": 2.0, "runner_objects": 2.0} and info["input_voxels"] == 6
    predictor.reset()
    assert all(fake.resets == 1 for fake in fakes)


# ---- config ----------------------------------------------------------------------------------------------------------
def test_config_section(tmp_path):
    text = (ROOT / "configs" / "difflow.yaml").read_text()
    section = _load_difflow_config(ROOT / "configs" / "difflow.yaml").object_flow
    assert section.mode == "joint" and section.context_cell_m == 0.15 and section.max_objects == 4
    assert section.scheduled.own_forwards == 2 and section.scheduled.budget_control and section.scheduled.multi_rate
    path = tmp_path / "difflow.yaml"
    path.write_text(text.replace("context_cell_m: 0.15", "context_cell_m: 0.2").replace("mode: joint", "mode: per_object"))
    section = _load_difflow_config(path).object_flow
    assert section.mode == "per_object" and section.context_cell_m == 0.2
    for good, bad in (
        ("mode: joint", "mode: both"), ("context_cell_m: 0.15", "context_cell_m: 0.0"), ("context_cell_m: 0.15", "context_cell_m: 0.15\n  cell: 1"),
        ("max_objects: 4", "max_objects: 0"), ("own_forwards: 2", "own_forwards: 0"), ("budget_control: true", "budget_control: 1"),
        ("target_cycle_ms: 31.0", "target_cycle_ms: -1"), ("multi_rate: true", "multi_rate: true\n    rate: 2"),
    ):
        assert good in text
        path.write_text(text.replace(good, bad))
        with pytest.raises(ValueError):
            _load_difflow_config(path)


def test_scheduled_needs_canonical_normalisation(tmp_path):
    text = (ROOT / "configs" / "difflow.yaml").read_text()
    path = tmp_path / "difflow.yaml"
    path.write_text(text.replace("mode: joint", "mode: scheduled"))
    if not _load_difflow_config(ROOT / "configs" / "difflow.yaml").preprocessing.canonical_normalization.enable:
        with pytest.raises(ValueError, match="canonical"):
            _load_difflow_config(path)
