import torch

from scene_pred_pipeline.voxel import voxel_downsample, voxel_iou


def test_voxel_downsample_is_stable():
    points = torch.tensor(
        [[0.001, 0.001, 0.001], [0.009, 0.009, 0.009], [0.021, 0.0, 0.0]]
    )
    origin = torch.zeros(3)
    sampled, keys = voxel_downsample(points, 0.01, origin)
    assert sampled.shape[0] == 2
    assert keys.shape[0] == 2
    assert float(voxel_iou(keys, keys)) == 1.0
