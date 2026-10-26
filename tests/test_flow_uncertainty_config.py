"""Flow-uncertainty configuration, calibration scale lookup and DifFlowPredictor wiring.

The DifFlow3D runner/head are replaced by fakes, so this runs without the
uncertainty-capable DifFlow3D runtime, checkpoints or a GPU.
"""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import yaml

from scene_pred_pipeline import flow_prediction
from scene_pred_pipeline.config import (
    DifFlowModelConfig,
    check_uncertainty_head_meta,
    load_config,
    load_config_from_mapping,
)
from scene_pred_pipeline.data_types import InstancePair
from scene_pred_pipeline.flow_prediction import (
    AnchorCalibrationScale,
    DifFlowPredictor,
    _build_model,
)

CONFIGS = Path(__file__).resolve().parents[1] / "configs"

# Tables of the umbrella configs/difflow_zu.yaml (real2 stab30 sel split, deployed path).
EDGES = [2.0, 10.0, 20.0, 40.0]
TABLES = {
    "ellipsoid": {
        "global": {0.5: 1.0096, 0.8: 1.7084, 0.95: 2.6101, 0.99: 3.5311, 0.999: 5.0408},
        "mondrian": {
            0.5: [0.7360, 1.3401, 0.9782, 0.9990, 1.6863],
            0.8: [1.2260, 2.0845, 1.5142, 1.6107, 3.2210],
            0.95: [1.9642, 2.9678, 2.2327, 2.5967, 5.6541],
            0.99: [2.7853, 3.8812, 2.9755, 3.7714, 7.6293],
            0.999: [4.3727, 5.3390, 4.0133, 5.5700, 9.0906],
        },
    },
    "ball": {
        "global": {0.5: 1.1525, 0.8: 2.0813, 0.95: 3.3250, 0.99: 4.5707, 0.999: 6.5428},
        "mondrian": {
            0.5: [0.8028, 1.5776, 1.1603, 1.1790, 2.0947],
            0.8: [1.4005, 2.5909, 1.9175, 2.0600, 4.7648],
            0.95: [2.3316, 3.8261, 2.9568, 3.5715, 8.1310],
            0.99: [3.3291, 4.9810, 4.0646, 5.5027, 11.1106],
            0.999: [5.1748, 6.8352, 5.6545, 8.3090, 13.7886],
        },
    },
}
MONDRIAN = TABLES["ellipsoid"]["mondrian"]


def _difflow_raw() -> dict:
    with (CONFIGS / "difflow.yaml").open() as stream:
        return yaml.safe_load(stream)


def _tables_raw() -> dict:
    return {
        region: {
            "global": {"scales": dict(table["global"])},
            "mondrian": {"edges_mm": list(EDGES), "scales": copy.deepcopy(table["mondrian"])},
        }
        for region, table in TABLES.items()
    }


def _uncertainty_raw(region="ellipsoid", mode="mondrian", level=0.95) -> dict:
    return {
        "enable": True,
        "checkpoint": "/weights/head.pt",
        "calibration": {
            "region": region,
            "mode": mode,
            "level": level,
            "source": "unit test",
            "tables": _tables_raw(),
        },
    }


def _zu_raw() -> dict:
    raw = _difflow_raw()
    raw["preprocessing"]["auto_spatial_scale"]["enable"] = False
    raw["preprocessing"]["canonical_normalization"] = {
        "enable": True,
        "model_center": [-0.52, -1.57, 21.36],
        "model_radius": 8.97,
        "scale_ratio": 1.5,
        "center_shift": 0.3,
    }
    raw["uncertainty_head"] = _uncertainty_raw()
    return raw


def _load(tmp_path, difflow_raw, **output):
    path = tmp_path / "difflow.yaml"
    path.write_text(yaml.safe_dump(difflow_raw))
    mapping = {
        "ros": {"camera_names": ["camera_0"]},
        "flow": {"config_path": str(path), "repo_path": str(tmp_path)},
        "output": output,
    }
    return load_config_from_mapping(mapping, base_dir=tmp_path)


# ---------------------------------------------------------------- config parsing
def test_existing_config_loads_with_feature_off():
    config = load_config(CONFIGS / "default.yaml")
    assert not config.difflow.uncertainty_head.enable
    assert not config.difflow.preprocessing.canonical_normalization.enable
    assert config.difflow.model.level1_points == 1024
    # On by default; without the head no marker topic is created (test_flow_uncertainty_markers).
    assert config.output.publish_uncertainty_markers
    assert config.output.uncertainty_marker_max == 150
    assert config.output.uncertainty_marker_publish_hz == pytest.approx(10.0)
    assert config.output.uncertainty_marker_alpha == pytest.approx(0.35)


def test_old_difflow_yaml_loads_with_feature_off(tmp_path):
    config = _load(tmp_path, _difflow_raw())
    assert config.difflow.preprocessing.auto_spatial_scale.enable
    assert not config.difflow.preprocessing.canonical_normalization.enable
    assert not config.difflow.uncertainty_head.enable
    # Deployment defaults of the calibration selection (used once a head is enabled).
    calibration = config.difflow.uncertainty_head.calibration
    assert (calibration.region, calibration.mode, calibration.level) == ("ellipsoid", "mondrian", 0.95)


def test_markers_on_without_head_load(tmp_path):
    # Markers on without a head load fine; RosVisualizer then creates no publisher.
    config = _load(tmp_path, _difflow_raw(), publish_uncertainty_markers=True)
    assert config.output.publish_uncertainty_markers and not config.difflow.uncertainty_head.enable


def test_canonical_and_uncertainty_head_load(tmp_path):
    config = _load(
        tmp_path,
        _zu_raw(),
        publish_uncertainty_markers=True,
        uncertainty_marker_max=50,
        uncertainty_marker_publish_hz=2.5,
    )
    canonical = config.difflow.preprocessing.canonical_normalization
    assert canonical.enable and canonical.model_center == (-0.52, -1.57, 21.36)
    head = config.difflow.uncertainty_head
    assert head.enable and head.checkpoint == "/weights/head.pt"
    calibration = head.calibration
    assert (calibration.region, calibration.mode) == ("ellipsoid", "mondrian")
    assert calibration.source == "unit test"
    for region in ("ellipsoid", "ball"):
        table = getattr(calibration, region)
        assert table.mondrian_edges_mm == tuple(EDGES)
        assert dict(table.global_scales) == TABLES[region]["global"]
        assert {level: list(row) for level, row in table.mondrian_scales} == TABLES[region]["mondrian"]
    assert calibration.table() is calibration.ellipsoid
    assert calibration.mondrian_bin_scales() == tuple(MONDRIAN[0.95])
    assert calibration.global_scale() == pytest.approx(TABLES["ellipsoid"]["global"][0.95])
    assert config.output.publish_uncertainty_markers
    assert config.output.uncertainty_marker_max == 50
    assert config.output.uncertainty_marker_publish_hz == pytest.approx(2.5)


@pytest.mark.parametrize("region", ["ellipsoid", "ball"])
@pytest.mark.parametrize("level", [0.5, 0.99])
def test_region_selects_its_tables(tmp_path, region, level):
    raw = _zu_raw()
    raw["uncertainty_head"]["calibration"].update(region=region, level=level)
    calibration = _load(tmp_path, raw).difflow.uncertainty_head.calibration
    assert calibration.table() is getattr(calibration, region)
    assert calibration.global_scale() == pytest.approx(TABLES[region]["global"][level])
    assert calibration.mondrian_bin_scales() == tuple(TABLES[region]["mondrian"][level])


def test_canonical_with_auto_spatial_scale_is_rejected(tmp_path):
    raw = _zu_raw()
    raw["preprocessing"]["auto_spatial_scale"]["enable"] = True
    with pytest.raises(ValueError, match="replaces auto_spatial_scale"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    "key, value, match",
    [
        ("model_radius", 0.0, "model_radius"),
        ("scale_ratio", 1.0, "scale_ratio"),
        ("center_shift", -0.1, "center_shift"),
        ("model_center", [0.0, 1.0], "model_center"),
    ],
)
def test_invalid_canonical_values(tmp_path, key, value, match):
    raw = _zu_raw()
    raw["preprocessing"]["canonical_normalization"][key] = value
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    "edit, name",
    [
        (lambda r: r["preprocessing"]["canonical_normalization"].update(radius=1.0),
         "difflow.preprocessing.canonical_normalization"),
        (lambda r: r["uncertainty_head"].update(head="U3"), "difflow.uncertainty_head"),
        # Old flat layout: tables now live under calibration.tables.<region>.
        (lambda r: r["uncertainty_head"]["calibration"].update(global_scales={0.95: 1.0}),
         "difflow.uncertainty_head.calibration"),
        (lambda r: r["uncertainty_head"]["calibration"].update(ellipsoid={}),
         "difflow.uncertainty_head.calibration"),
        (lambda r: r["uncertainty_head"]["calibration"]["tables"].update(sphere={}),
         "difflow.uncertainty_head.calibration.tables"),
        (lambda r: r["uncertainty_head"]["calibration"]["tables"]["ball"].update(scales={}),
         "difflow.uncertainty_head.calibration.tables.ball"),
        (lambda r: r["uncertainty_head"]["calibration"]["tables"]["ellipsoid"]["mondrian"].update(bins=[1.0]),
         "difflow.uncertainty_head.calibration.tables.ellipsoid.mondrian"),
        (lambda r: r["uncertainty_head"]["calibration"]["tables"]["ball"]["global"].update(levels=[0.9]),
         "difflow.uncertainty_head.calibration.tables.ball.global"),
    ],
)
def test_unknown_keys_raise_value_error(tmp_path, edit, name):
    raw = _zu_raw()
    edit(raw)
    with pytest.raises(ValueError, match=f"Unknown keys in {name.replace('.', '[.]')}:"):
        _load(tmp_path, raw)


@pytest.mark.parametrize("region", ["ellipsoid", "ball"])
@pytest.mark.parametrize("mode", ["global", "mondrian"])
def test_level_must_be_in_selected_table(tmp_path, region, mode):
    raw = _zu_raw()
    raw["uncertainty_head"]["calibration"].update(region=region, mode=mode, level=0.9)
    with pytest.raises(ValueError, match=f"not a level of the {region} {mode} table"):
        _load(tmp_path, raw)


def test_level_only_needs_the_selected_table(tmp_path):
    raw = _zu_raw()
    tables = raw["uncertainty_head"]["calibration"]["tables"]
    del tables["ball"]
    del tables["ellipsoid"]["mondrian"]
    raw["uncertainty_head"]["calibration"]["mode"] = "global"
    config = _load(tmp_path, raw)
    assert config.difflow.uncertainty_head.calibration.global_scale() == pytest.approx(2.6101)
    raw["uncertainty_head"]["calibration"]["mode"] = "mondrian"
    with pytest.raises(ValueError, match="not a level"):
        _load(tmp_path, raw)
    raw["uncertainty_head"]["calibration"].update(mode="global", region="ball")
    with pytest.raises(ValueError, match="not a level of the ball global table"):
        _load(tmp_path, raw)


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda c: c.update(mode="conformal"), "global or mondrian"),
        (lambda c: c.update(region="sphere"), "ellipsoid or ball"),
        (lambda c: c.update(level=1.0), "coverage level"),
        (lambda c: c.update(level=True), "number"),
        (lambda c: c["tables"]["ellipsoid"]["mondrian"]["scales"][0.95].pop(), "5 scales"),
        (lambda c: c["tables"]["ball"]["mondrian"]["scales"].update({0.9: 1.0}), "5 scales"),
        (lambda c: c["tables"]["ball"]["mondrian"].update(edges_mm=[2.0, 2.0, 20.0, 40.0]),
         "strictly increasing"),
        (lambda c: c["tables"]["ellipsoid"]["mondrian"].update(edges_mm=[10.0, 2.0, 20.0, 40.0]),
         "strictly increasing"),
        (lambda c: c["tables"]["ellipsoid"]["global"]["scales"].update({0.95: -1.0}), "positive"),
        (lambda c: c["tables"]["ball"]["global"]["scales"].update({1.5: 1.0}), "coverage level"),
    ],
)
def test_invalid_calibration_tables(tmp_path, edit, match):
    raw = _zu_raw()
    edit(raw["uncertainty_head"]["calibration"])
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, raw)


def test_enabled_head_needs_checkpoint(tmp_path):
    raw = _zu_raw()
    raw["uncertainty_head"]["checkpoint"] = ""
    with pytest.raises(ValueError, match="checkpoint"):
        _load(tmp_path, raw)


def test_disabled_head_ignores_level(tmp_path):
    raw = _difflow_raw()
    raw["uncertainty_head"] = {"enable": False, "calibration": {"level": 0.9}}
    assert not _load(tmp_path, raw).difflow.uncertainty_head.enable


@pytest.mark.parametrize(
    "output, match",
    [
        ({"uncertainty_marker_max": 0}, "uncertainty_marker_max"),
        ({"uncertainty_marker_max": True}, "uncertainty_marker_max"),
        ({"uncertainty_marker_alpha": 0.0}, "uncertainty_marker_alpha"),
        ({"uncertainty_marker_alpha": 1.5}, "uncertainty_marker_alpha"),
        ({"uncertainty_marker_alpha": True}, "uncertainty_marker_alpha must be a number"),
        ({"uncertainty_marker_alpha": "opaque"}, "uncertainty_marker_alpha must be a number"),
        ({"uncertainty_marker_publish_hz": 0.0}, "uncertainty_marker_publish_hz must be positive"),
        ({"uncertainty_marker_publish_hz": -5.0}, "uncertainty_marker_publish_hz must be positive"),
        ({"uncertainty_marker_publish_hz": False}, "uncertainty_marker_publish_hz must be a number"),
        ({"uncertainty_marker_publish_hz": "fast"}, "uncertainty_marker_publish_hz must be a number"),
        ({"uncertainty_marker_publish_hz": float("inf")}, "uncertainty_marker_publish_hz must be finite"),
        ({"publish_uncertainty_markers": "yes"}, "boolean"),
    ],
)
def test_invalid_output_keys(tmp_path, output, match):
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, _difflow_raw(), **output)


# ---------------------------------------------------------------- model level 1
def _with_model(raw, **model):
    raw["model"].update(model)
    return raw


def test_level1_points_parsing(tmp_path):
    raw = _difflow_raw()
    assert "level1_points" not in raw["model"]
    assert _load(tmp_path, raw).difflow.model.level1_points == 1024
    # Buckets 1024 (level 1 = the input itself) and >= 2048 are valid with 2048.
    raw = _with_model(_difflow_raw(), level1_points=2048)
    raw["preprocessing"].update(fps_points=4096, point_buckets=[1024, 2048, 3072, 4096])
    assert _load(tmp_path, raw).difflow.model.level1_points == 2048
    # Without buckets the runner uses fps_points only.
    raw = _with_model(_difflow_raw(), level1_points=4096)
    raw["preprocessing"].update(fps_points=4096, point_buckets=None)
    assert _load(tmp_path, raw).difflow.model.level1_points == 4096


@pytest.mark.parametrize(
    "level1_points, buckets, fps_points, match",
    [
        (512, [1024, 2048], 2048, "integer >= 1024"),
        (True, [1024, 2048], 2048, "integer >= 1024"),
        (2048.0, [1024, 2048], 2048, "integer >= 1024"),
        ("2048", [1024, 2048], 2048, "integer >= 1024"),
        (3072, [1024, 2048, 3072, 4096], 4096, r"buckets \[2048\] are smaller"),
        (8192, None, 4096, r"buckets \[4096\] are smaller"),
    ],
)
def test_invalid_level1_points(tmp_path, level1_points, buckets, fps_points, match):
    raw = _with_model(_difflow_raw(), level1_points=level1_points)
    raw["preprocessing"].update(fps_points=fps_points, point_buckets=buckets)
    with pytest.raises(ValueError, match=match):
        _load(tmp_path, raw)


class _ModelWithLevel1:
    def __init__(self, iters, coarse_iters, middle_iters, fine_iters, level1_points=1024):
        self.kwargs = dict(
            iters=iters,
            coarse_iters=coarse_iters,
            middle_iters=middle_iters,
            fine_iters=fine_iters,
            level1_points=level1_points,
        )


class _LegacyModel:
    def __init__(self, iters, coarse_iters, middle_iters, fine_iters):
        self.kwargs = dict(
            iters=iters, coarse_iters=coarse_iters, middle_iters=middle_iters, fine_iters=fine_iters
        )


@pytest.mark.parametrize("level1_points", [1024, 2048])
def test_model_gets_level1_points(level1_points):
    model = _build_model(_ModelWithLevel1, DifFlowModelConfig(level1_points=level1_points))
    assert model.kwargs == dict(
        iters=4, coarse_iters=4, middle_iters=2, fine_iters=2, level1_points=level1_points
    )


def test_legacy_model_without_level1_points():
    # Older DifFlow3D checkouts (level 1 fixed at 1024) keep working at the default.
    assert _build_model(_LegacyModel, DifFlowModelConfig()).kwargs["iters"] == 4
    with pytest.raises(RuntimeError, match="level1_points"):
        _build_model(_LegacyModel, DifFlowModelConfig(level1_points=2048))


# ---------------------------------------------------------------- head meta checks
def test_head_meta_iterations_must_match(tmp_path):
    difflow = _load(tmp_path, _zu_raw()).difflow
    check_uncertainty_head_meta({"iters": "4,2,2"}, difflow)
    check_uncertainty_head_meta({"iters": [4, 2, 2]}, difflow)
    with pytest.raises(ValueError, match="iterations"):
        check_uncertainty_head_meta({"iters": "4,4,4"}, difflow)
    with pytest.raises(ValueError, match="iters"):
        check_uncertainty_head_meta({}, difflow)
    with pytest.raises(ValueError, match="iters"):
        check_uncertainty_head_meta({"iters": "four"}, difflow)


def test_canonical_head_needs_canonical_normalization(tmp_path):
    canonical = _load(tmp_path, _zu_raw()).difflow
    raw = _zu_raw()
    raw["preprocessing"]["canonical_normalization"]["enable"] = False
    raw["preprocessing"]["auto_spatial_scale"]["enable"] = True
    plain = _load(tmp_path, raw).difflow
    meta = {"iters": "4,2,2", "canonical": True}
    check_uncertainty_head_meta(meta, canonical)
    with pytest.raises(ValueError, match="canonically normalised"):
        check_uncertainty_head_meta(meta, plain)
    # A head without the flag (or canonical: False) does not constrain the preprocessing.
    check_uncertainty_head_meta({"iters": "4,2,2", "canonical": False}, plain)
    check_uncertainty_head_meta({"iters": "4,2,2"}, plain)


def _fake_difflow_modules(meta):
    """difflow3d.model / difflow3d.runtime stand-ins for DifFlowPredictor.__init__."""

    class Model(_ModelWithLevel1):
        def to(self, device):
            return self

        def eval(self):
            return self

        def modules(self):
            return []

        def configure_dense_points(self, **kwargs):
            self.dense = kwargs

    runtime = SimpleNamespace(
        DifFlow3DStreamingCudaGraphRunner=object,
        configure_fast_inference=lambda enable: None,
        load_checkpoint=lambda model, path, strict: SimpleNamespace(missing_keys=[], unexpected_keys=[]),
        resolve_voxel_outlier_statistics=None,
        load_flow_covariance_head=lambda path, device: (torch.nn.Identity(), dict(meta)),
    )
    return {"difflow3d.model": SimpleNamespace(PointConvBidirection=Model), "difflow3d.runtime": runtime}


def _predictor_with_fakes(monkeypatch, config, meta):
    modules = _fake_difflow_modules(meta)
    real_import = flow_prediction.importlib.import_module
    monkeypatch.setattr(
        flow_prediction.importlib,
        "import_module",
        lambda name, *args: modules[name] if name in modules else real_import(name, *args),
    )
    return DifFlowPredictor(config, upstream_voxel_size_m=0.005)


def test_predictor_rejects_canonical_head_without_normalization(tmp_path, monkeypatch):
    raw = _zu_raw()
    raw["preprocessing"]["canonical_normalization"]["enable"] = False
    raw["preprocessing"]["auto_spatial_scale"]["enable"] = True
    config = _load(tmp_path, raw)
    with pytest.raises(ValueError, match="canonically normalised"):
        _predictor_with_fakes(monkeypatch, config, {"iters": "4,2,2", "canonical": True})


def test_predictor_passes_level1_points(tmp_path, monkeypatch):
    raw = _with_model(_difflow_raw(), level1_points=2048)
    raw["preprocessing"].update(fps_points=4096, point_buckets=[1024, 2048, 3072, 4096])
    predictor = _predictor_with_fakes(monkeypatch, _load(tmp_path, raw), {})
    assert predictor.model.kwargs["level1_points"] == 2048
    assert predictor.uncertainty_head is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="calibration tables live on CUDA")
def test_predictor_loads_canonical_head(tmp_path, monkeypatch):
    predictor = _predictor_with_fakes(
        monkeypatch, _load(tmp_path, _zu_raw()), {"iters": "4,2,2", "canonical": True}
    )
    assert predictor.uncertainty_head is not None
    assert predictor.uncertainty_head_meta["canonical"] is True
    assert predictor.calibration_scale.region == "ellipsoid"
    assert predictor.model.kwargs["level1_points"] == 1024


# ---------------------------------------------------------------- calibration scale
def _calibration(tmp_path, mode, level=0.95, region="ellipsoid"):
    raw = _zu_raw()
    raw["uncertainty_head"]["calibration"].update(region=region, mode=mode, level=level)
    return _load(tmp_path, raw).difflow.uncertainty_head.calibration


def test_mondrian_bins_include_lower_edge(tmp_path):
    scale = AnchorCalibrationScale(_calibration(tmp_path, "mondrian"), "cpu")
    displacement_mm = torch.tensor(
        [0.0, 1.999, 2.0, 9.999, 10.0, 19.999, 20.0, 39.999, 40.0, 500.0]
    )
    expected_bins = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    expected = torch.tensor([MONDRIAN[0.95][b] for b in expected_bins])
    assert torch.equal(scale.for_displacement_mm(displacement_mm), expected)


def test_mondrian_scale_from_world_flow(tmp_path):
    scale = AnchorCalibrationScale(_calibration(tmp_path, "mondrian", level=0.5), "cpu")
    # |flow| = 1 mm, 5 mm (3-4-0 triangle), 15 mm, 30 mm, 0.1 m
    flow = torch.tensor(
        [[0.001, 0.0, 0.0], [0.003, 0.004, 0.0], [0.0, 0.015, 0.0], [0.0, 0.0, -0.03], [0.1, 0.0, 0.0]]
    )
    assert torch.equal(scale(flow), torch.tensor(MONDRIAN[0.5]))


def test_global_scale_is_constant(tmp_path):
    scale = AnchorCalibrationScale(_calibration(tmp_path, "global", level=0.99), "cpu")
    values = scale(torch.randn(7, 3))
    expected = torch.tensor(TABLES["ellipsoid"]["global"][0.99])
    assert values.shape == (7,) and torch.all(values == expected)


@pytest.mark.parametrize("mode", ["global", "mondrian"])
def test_ball_region_uses_ball_tables(tmp_path, mode):
    scale = AnchorCalibrationScale(_calibration(tmp_path, mode, region="ball"), "cpu")
    assert (scale.region, scale.mode) == ("ball", mode)
    # |flow| = 1, 5, 15, 30, 100 mm: one point per Mondrian bin.
    flow = torch.tensor(
        [[0.001, 0.0, 0.0], [0.003, 0.004, 0.0], [0.0, 0.015, 0.0], [0.0, 0.0, -0.03], [0.1, 0.0, 0.0]]
    )
    table = TABLES["ball"][mode][0.95]
    expected = torch.tensor(table if mode == "mondrian" else [table] * 5)
    assert torch.equal(scale(flow), expected)


# ---------------------------------------------------------------- predictor wiring
class _FakeRunner:
    """Records the constructor arguments; serves fixed world outputs after two frames."""

    created: list[dict] = []

    def __init__(self, model, **kwargs):
        _FakeRunner.created.append(kwargs)
        self.has_uncertainty = kwargs.get("uncertainty_head") is not None
        self.frames = 0
        self.flow = torch.tensor([[[0.001, 0.0, 0.0], [0.0, 0.025, 0.0], [0.0, 0.0, 0.05]]])
        self.points = torch.zeros((1, 3, 3))
        self.cov = torch.eye(3).expand(1, 3, 3, 3) * 1.0e-6

    def reset(self):
        self.frames = 0

    def begin_profile_window(self):
        pass

    def stage_world(self, points, point_ids=None):
        self.ids = point_ids
        return points

    def replay_next(self):
        self.frames += 1
        return None if self.frames == 1 else object()

    def source_points_world(self):
        return self.points

    def warped_points_world(self):
        return self.points + self.flow

    def flow_world(self):
        return self.flow

    def flow_cov_world(self):
        return self.cov.clone()

    def source_point_ids(self):
        return torch.tensor([3, 3, 4])

    def target_input_keep_mask(self):
        return torch.ones(3, dtype=torch.bool)

    def target_preprocess_info(self):
        return {}


def _predictor(config, *, head):
    predictor = DifFlowPredictor.__new__(DifFlowPredictor)
    predictor.config = config
    predictor.model = object()
    predictor.runner = None
    predictor.runner_class = _FakeRunner
    predictor.upstream_voxel_size_m = 0.01
    predictor.uncertainty_head = head
    predictor.calibration_scale = (
        AnchorCalibrationScale(config.difflow.uncertainty_head.calibration, "cpu")
        if head is not None
        else None
    )
    predictor._cached_target_stamp_ns = None
    predictor._cached_track_signature = None
    predictor._last_target_preprocess_info = None
    predictor._debug_pending_current = None
    return predictor


def _pair():
    points = torch.zeros((3, 3))
    ids = torch.tensor([3, 3, 4], dtype=torch.int32)
    return InstancePair(
        previous_stamp_ns=0,
        current_stamp_ns=33_000_000,
        common_track_ids=(3, 4),
        previous_points=points,
        current_points=points,
        previous_track_ids=ids,
        current_track_ids=ids,
        dt_s=0.033,
    )


def test_runner_gets_normalization_and_head(tmp_path):
    pytest.importorskip("difflow3d.runtime.normalization")
    from difflow3d.runtime.normalization import AnchoredNormalization

    config = _load(tmp_path, _zu_raw())
    head = torch.nn.Identity()
    predictor = _predictor(config, head=head)
    predictor._ensure_runner()
    kwargs = _FakeRunner.created[-1]
    normalization = kwargs["normalization"]
    assert isinstance(normalization, AnchoredNormalization)
    assert normalization.model_center == (-0.52, -1.57, 21.36)
    assert normalization.model_radius == pytest.approx(8.97)
    assert normalization.scale_ratio == pytest.approx(1.5)
    assert normalization.center_shift == pytest.approx(0.3)
    assert kwargs["auto_spatial_scale"] is False
    assert kwargs["uncertainty_head"] is head

    result = predictor.predict(_pair())
    assert result.anchor_flow_cov.shape == (3, 3, 3)
    assert torch.allclose(result.anchor_flow_cov, torch.eye(3).expand(3, 3, 3) * 1.0e-6)
    # |flow| = 1, 25, 50 mm -> Mondrian bins <2, 20-40, 40+ at level 0.95
    expected = torch.tensor([MONDRIAN[0.95][0], MONDRIAN[0.95][3], MONDRIAN[0.95][4]])
    assert torch.equal(result.anchor_flow_cov_scale, expected)


def test_runner_without_feature_gets_no_new_arguments(tmp_path):
    predictor = _predictor(_load(tmp_path, _difflow_raw()), head=None)
    predictor._ensure_runner()
    kwargs = _FakeRunner.created[-1]
    assert "normalization" not in kwargs and "uncertainty_head" not in kwargs
    assert kwargs["auto_spatial_scale"] is True
    result = predictor.predict(_pair())
    assert result.anchor_flow_cov is None and result.anchor_flow_cov_scale is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="pipeline buffers live on CUDA")
def test_pipeline_carries_uncertainty_to_output(tmp_path):
    from scene_pred_pipeline.data_types import FlowResult, TrackedInstanceFrame
    from scene_pred_pipeline.pipeline import ScenePredictionPipeline
    from scene_pred_pipeline.profiler import CycleProfiler

    config = _load(tmp_path, _zu_raw())
    cov = torch.eye(3).expand(4, 3, 3) * 4.0e-6
    cov_scale = torch.full((4,), 2.5)
    anchor_ids = torch.tensor([3, 3, 4, 4])
    flow_result = FlowResult(
        source_anchors=torch.zeros((4, 3)),
        warped_anchors=torch.ones((4, 3)),
        anchor_flow=torch.ones((4, 3)),
        source_anchor_track_ids=anchor_ids,
        target_input_keep_mask=torch.ones(3, dtype=torch.bool),
        anchor_flow_cov=cov,
        anchor_flow_cov_scale=cov_scale,
    )
    tracked = TrackedInstanceFrame(
        frame_index=1,
        stamp_ns=33_000_000,
        instances=[],
        packed_points_world=torch.zeros((3, 3)),
        packed_track_ids=torch.tensor([3, 3, 4], dtype=torch.int32),
        track_ids=(3, 4),
        track_offsets={},
    )
    pipeline = ScenePredictionPipeline.__new__(ScenePredictionPipeline)
    pipeline.config = config
    pipeline.static_only = False
    pipeline.tracker = SimpleNamespace(process=lambda frame: tracked)
    pipeline.instance_filter = SimpleNamespace(select=lambda previous, current: _pair())
    pipeline.flow_predictor = SimpleNamespace(
        detailed_outlier_output=False,
        outlier_filter_enabled=False,
        predict=lambda pair, track_labels=None: flow_result,
        resolve_after_cycle_fence=lambda: ({}, {}, {}),
        reset=lambda: None,
    )
    pipeline.recovery = SimpleNamespace(
        recover=lambda pair, result, query_points, query_track_ids: torch.zeros_like(query_points)
    )
    pipeline.scene_assembler = None
    pipeline.profiler = CycleProfiler(False)
    pipeline.previous_tracked = None
    pipeline.last_flow_gap_s = None

    output = pipeline.process(SimpleNamespace(stamp_ns=33_000_000))
    assert output.flow_valid
    assert output.anchor_flow_cov is cov and output.anchor_flow_cov_scale is cov_scale
    assert output.anchor_track_ids is anchor_ids
