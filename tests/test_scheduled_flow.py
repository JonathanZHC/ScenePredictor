"""object_flow.mode scheduled: planning, budget, row state and the cycle (CPU, fake runners and row batches)."""
import pytest
import torch

from flow_fakes import DT, FakePredictor, FakeRowBatch, blob, config, pair
from scene_pred_pipeline.scheduled_flow import (
    BudgetController,
    RowTrack,
    ScheduledFlowPredictor,
    due_rows,
    own_groups,
    plan_objects,
    update_period,
)


def _predictor(**scheduled):
    fakes, batches = [], []

    def factory():
        fakes.append(FakePredictor())
        return fakes[-1]

    def row_batch(anchors, row_counts):
        batches.append(FakeRowBatch(anchors, row_counts))
        return batches[-1]

    predictor = ScheduledFlowPredictor(
        config(mode="scheduled", scheduled=scheduled), upstream_voxel_size_m=0.005, predictor_factory=factory, row_batch_factory=row_batch
    )
    predictor.prepare()
    return predictor, fakes, batches


# ---- planning --------------------------------------------------------------------------------------------------------
def _plan(extent, touching, peak_speed, own_forwards=2, count=None):
    objects = sorted(extent)
    count = count or {track: 1000 for track in objects}
    return plan_objects(objects, count, extent, {track: track in touching for track in objects}, peak_speed, own_forwards)


def test_isolated_objects_are_rows_by_size():
    own, rows = _plan({1: 0.2, 2: 0.35, 3: 0.6, 4: 0.9}, touching=(), peak_speed={})
    assert own == [] and rows == [[1, 2], [3, 4]]


def test_touching_and_large_objects_get_own_forwards_fastest_first():
    extent = {1: 1.7, 2: 0.6, 3: 0.3, 4: 0.3, 5: 1.0}
    own, rows = _plan(extent, touching=(1, 2, 3, 4), peak_speed={1: 0.5, 2: 3.0, 3: 0.1, 4: 0.2, 5: 0.0})
    assert own == [2, 1]                                 # the two fastest of the touching / large ones
    assert rows == [[3, 4], [5]]                         # the others are rows: touching does not force a forward


def test_an_object_too_large_for_a_row_keeps_a_forward():
    own, rows = _plan({1: 0.3, 2: 0.3, 3: 1.7}, touching=(1, 2, 3), peak_speed={1: 2.0, 2: 1.5, 3: 0.0})
    assert own == [1, 2, 3] and rows == [[], []]
    assert own_groups(own, 2) == [(1,), (2, 3)] and own_groups(own, 1) == [(1, 2, 3)] and own_groups([4], 2) == [(4,)]


def test_new_objects_come_first_and_sparse_ones_are_rows_or_dropped():
    own, rows = _plan({1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}, touching=(1, 2, 3, 4), peak_speed={1: 1.0, 2: 2.0},
                      own_forwards=1, count={1: 1000, 2: 1000, 3: 1000, 4: 10})
    assert own == [3] and rows == [[], [1, 2]]           # 3 has no speed yet; 4 has too few points for any estimate


def test_the_period_follows_speed_and_acceleration():
    assert [update_period(speed, 0.0, True) for speed in (2.0, 0.5, 0.1)] == [1, 2, 3]
    assert [update_period(0.0, accel, True) for accel in (4.0, 2.0, 0.5)] == [1, 2, 3]
    assert update_period(0.0, 0.0, False) == 1


def test_due_rows_are_new_first_then_most_overdue():
    track = lambda cycle, period: RowTrack(torch.zeros(4, 3), 0, cycle, torch.zeros(4, 3), torch.zeros(4, 3), None, torch.zeros(3), 0.0, DT, torch.zeros(3), 0.0, period)
    tracks = {1: track(8, 3), 2: track(7, 1), 3: track(9, 2)}
    assert due_rows([1, 2, 3, 4], tracks, cycle=10, limit=9, anchors=4) == [4, 2]         # 1 and 3 are not due yet
    assert due_rows([1, 2, 3, 4], tracks, cycle=11, limit=2, anchors=4) == [4, 2]
    assert due_rows([1, 2, 3], tracks, cycle=11, limit=9, anchors=8) == [3, 2, 1]         # another row class: all new


# ---- budget ----------------------------------------------------------------------------------------------------------
def _run(budget, cycle_ms, cycles):
    for _ in range(cycles):
        budget.observe(cycle_ms)


def test_the_budget_postpones_rows_before_it_merges_forwards():
    budget = BudgetController(31.0, True)
    _run(budget, 30.0, 100)
    assert budget.index == 0 and budget.median_ms == 30.0
    _run(budget, 32.0, 500)                              # over the target, but not by 3 ms: stays in the normal levels
    assert budget.index == BudgetController.NORMAL - 1 and not budget.level.merge_own
    _run(budget, 36.0, 500)
    assert budget.index == len(BudgetController.LEVELS) - 1 and budget.level.merge_own
    _run(budget, 20.0, 1000)
    assert budget.index == 0


def test_a_single_slow_cycle_changes_nothing_and_a_disabled_budget_stays():
    budget = BudgetController(31.0, True)
    _run(budget, 25.0, 30)
    budget.observe(90.0)
    _run(budget, 25.0, 30)
    assert budget.index == 0
    disabled = BudgetController(31.0, False)
    _run(disabled, 50.0, 200)
    assert disabled.index == 0 and disabled.median_ms == 50.0


# ---- row state -------------------------------------------------------------------------------------------------------
def test_a_row_is_carried_with_its_acceleration():
    velocity = torch.tensor([[1.0, 0.0, 0.0]])
    track = RowTrack(anchors=torch.zeros(1, 3), stamp_ns=0, cycle=1, position=torch.zeros(1, 3), velocity=velocity, cov=None,
                     mean_velocity=velocity[0], centre_s=-0.5 * DT, span_s=DT, accel=torch.tensor([3.0, 0.0, 0.0]), accel_ms2=3.0)
    source, position, flow, growth_ms = track.carried(int(round(2 * DT * 1e9)), DT)
    now_velocity = 1.0 + 3.0 * 2.5 * DT                              # the estimate belongs to half a frame before its stamp
    assert flow[0, 0].item() == pytest.approx(now_velocity * DT, rel=1e-5)
    assert position[0, 0].item() == pytest.approx(2 * DT * now_velocity, rel=1e-5) and torch.allclose(source, position - flow)
    assert growth_ms == pytest.approx(1.5 * 2.5 * DT, rel=1e-5)      # half the measured acceleration over the lead


# ---- the cycle -------------------------------------------------------------------------------------------------------
def _moving(clouds, velocity, frames):
    return {track: cloud + frames * DT * torch.tensor(velocity[track]) for track, cloud in clouds.items()}


def test_isolated_objects_are_rows_with_their_own_flow():
    predictor, fakes, (small, medium) = _predictor()
    clouds = {1: blob([0.0, 0.0, 1.0], 400, 0, spread=0.2), 2: blob([2.0, 0.0, 1.0], 300, 1, spread=0.6)}
    velocity = {1: [1.5, 0.0, 0.0], 2: [0.0, -1.2, 0.0]}
    result = predictor.predict(pair(_moving(clouds, velocity, 1), previous=clouds))
    assert not any(fake.calls for fake in fakes) and small.launched == [1] and medium.launched == [1]
    assert (result.source_anchor_track_ids == 1).sum() == 256 and (result.source_anchor_track_ids == 2).sum() == 512
    for track in (1, 2):
        flow = result.anchor_flow[result.source_anchor_track_ids == track]
        assert torch.allclose(flow, DT * torch.tensor(velocity[track]).expand_as(flow), atol=1e-5)
    assert torch.allclose(result.warped_anchors, result.source_anchors + result.anchor_flow, atol=1e-5)
    assert result.anchor_flow_cov.shape == (768, 3, 3) and result.target_input_keep_mask.all()
    assert predictor.resolve_after_cycle_fence()[0] == {                 # no runner ran: only what the schedule did
        "schedule_own_objects": 0.0, "schedule_row_objects": 2.0, "schedule_rows_estimated": 2.0, "schedule_budget_level": 0.0}


def test_touching_objects_get_own_forwards_with_context_and_the_rest_are_rows():
    predictor, fakes, (small, medium) = _predictor()
    stick = torch.stack([torch.linspace(0.0, 0.6, 300), torch.zeros(300), torch.ones(300)], 1)
    person = blob([-1.00, -0.50, 0.50], 600, 2, spread=1.00)
    ball = blob([3.0, 0.0, 1.0], 200, 3, spread=0.1)
    result = predictor.predict(pair({1: stick, 2: person, 3: ball}))
    calls = {fake.calls[0].common_track_ids[0]: fake.calls[0] for fake in fakes}
    assert set(calls) == {1, 2} and small.launched == [1] and medium.launched == []
    assert 0 < (calls[1].current_track_ids == 2).sum() < 600             # the stick's forward sees the person nearby
    assert not (calls[1].current_track_ids == 3).any() and not (calls[2].current_track_ids == 3).any()
    counts = {track: int((result.source_anchor_track_ids == track).sum()) for track in (1, 2, 3)}
    assert counts == {1: 300, 2: 600, 3: 256}                            # own anchors only, the ball's row
    timing = predictor.resolve_after_cycle_fence()[0]
    assert timing["runner_objects"] == 2.0 and timing["schedule_own_objects"] == 2.0 and timing["schedule_row_objects"] == 1.0


def test_a_slow_row_is_carried_between_its_estimates():
    predictor, _, (small, _) = _predictor()
    clouds = {1: blob([0.0, 0.0, 1.0], 400, 0, spread=0.2), 2: blob([2.0, 0.0, 1.0], 400, 1, spread=0.2)}
    velocity = {1: [2.0, 0.0, 0.0], 2: [0.0, 0.2, 0.0]}                  # fast: every cycle, slow: every third
    for frame in range(1, 8):
        result = predictor.predict(pair(_moving(clouds, velocity, frame), stamp=frame, previous_stamp=frame - 1,
                                        previous=_moving(clouds, velocity, frame - 1)))
        for track in (1, 2):                                             # exact whether fresh or carried
            own = result.source_anchor_track_ids == track
            assert torch.allclose(result.anchor_flow[own], DT * torch.tensor(velocity[track]).expand(256, 3), atol=1e-5)
        slow = result.warped_anchors[result.source_anchor_track_ids == 2]
        assert torch.allclose(slow.mean(0), _moving(clouds, velocity, frame)[2].mean(0), atol=0.02)
    assert small.launched == [2, 1, 1, 2, 1, 1, 2]
    carried = predictor._tracks[2]
    assert carried.period == 3 and carried.span_s == pytest.approx(3 * DT)      # a three-frame baseline


def test_the_uncertainty_of_a_carried_row_grows():
    predictor, _, _ = _predictor()
    clouds = {1: blob([0.0, 0.0, 1.0], 400, 0, spread=0.2)}
    volumes = []
    for frame in range(1, 4):
        result = predictor.predict(pair(clouds, stamp=frame, previous_stamp=frame - 1))
        volumes.append(float(torch.linalg.det(result.anchor_flow_cov).mean()))
    assert volumes[1] > volumes[0] and volumes[2] > volumes[1]
    assert torch.all(result.anchor_flow_cov_scale == 2.0)


def test_without_multi_rate_every_row_runs_every_cycle():
    predictor, _, (small, _) = _predictor(multi_rate=False)
    clouds = {1: blob([0.0, 0.0, 1.0], 400, 0, spread=0.2), 2: blob([2.0, 0.0, 1.0], 400, 1, spread=0.2)}
    for frame in range(1, 4):
        predictor.predict(pair(clouds, stamp=frame, previous_stamp=frame - 1))
    assert small.launched == [2, 2, 2]


def test_the_budget_limits_rows_and_merges_forwards():
    predictor, fakes, (small, _) = _predictor(multi_rate=False)
    balls = {track: blob([1.0 * track, 0.0, 1.0], 100, track, spread=0.1) for track in range(1, 7)}
    predictor.budget.index = 3                                           # three rows per cycle
    predictor.predict(pair(balls))
    predictor.predict(pair(balls, stamp=2, previous_stamp=1))
    assert small.launched == [3, 3]
    assert len(predictor._tracks) == 6                                   # the second cycle served the other three
    stick = torch.stack([torch.linspace(0.0, 0.6, 300), torch.zeros(300), torch.ones(300)], 1)
    person = blob([-1.00, -0.50, 0.50], 600, 2, spread=1.00)
    predictor.reset()
    predictor.budget.index = len(BudgetController.LEVELS) - 1            # overload: one shared, thinned forward
    result = predictor.predict(pair({1: stick, 2: person}))
    assert sum(len(fake.calls) for fake in fakes) == 1
    shared = next(fake.calls[0] for fake in fakes if fake.calls)
    assert 0.6 * 900 < shared.current_points.shape[0] < 0.85 * 900
    assert set(result.source_anchor_track_ids.tolist()) == {1, 2}


def test_reset_forgets_rows_and_the_pipeline_reports_its_cycle_time():
    predictor, fakes, _ = _predictor()
    predictor.predict(pair({1: blob([0.0, 0.0, 1.0], 400, 0, spread=0.2)}))
    assert predictor._tracks
    predictor.reset()
    assert not predictor._tracks and all(fake.resets == 1 for fake in fakes)
    for _ in range(BudgetController.WINDOW):
        predictor.observe_cycle_ms(40.0)
    assert predictor.budget.index == 1
