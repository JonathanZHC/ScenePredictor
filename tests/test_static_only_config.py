"""Empty tracker.tracked_prompts selects the static-only mode."""
from pathlib import Path

import pytest

from scene_pred_pipeline.config import load_config_from_mapping

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _mapping(**scene_cloud):
    return {
        "ros": {"camera_names": ["camera_0"]},
        "tracker": {"tracked_prompts": []},
        "flow": {"config_path": str(CONFIGS / "difflow.yaml")},
        "scene_cloud": scene_cloud,
    }


def test_static_only_requires_voxel_size():
    with pytest.raises(ValueError, match="voxel_size_m"):
        load_config_from_mapping(_mapping(), base_dir=CONFIGS)


def test_static_only_loads():
    cfg = load_config_from_mapping(_mapping(voxel_size_m=0.02), base_dir=CONFIGS)
    assert cfg.tracker.static_only and cfg.tracker.max_tracked_instances == 0
    assert cfg.scene_cloud.voxel_size_m == 0.02


def test_excluded_without_tracked_is_rejected():
    m = _mapping(voxel_size_m=0.02)
    m["tracker"]["excluded_prompts"] = [["human", 1]]
    with pytest.raises(ValueError, match="excluded_prompts"):
        load_config_from_mapping(m, base_dir=CONFIGS)


def test_tracked_mode_counts_capacity():
    m = _mapping(voxel_size_m=None)
    m["tracker"]["tracked_prompts"] = [["human", 1], ["chair", 2]]
    cfg = load_config_from_mapping(m, base_dir=CONFIGS)
    assert not cfg.tracker.static_only and cfg.tracker.max_tracked_instances == 3
