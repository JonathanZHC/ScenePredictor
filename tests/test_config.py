from scene_pred_pipeline.config import load_config


def test_default_config_loads():
    config = load_config("configs/default.yaml")
    assert config.ros.camera_names == ("camera_0", "camera_1")
    assert config.flow.target_points == 2048
