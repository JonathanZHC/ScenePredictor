import pytest
import torch

from scene_pred_pipeline.config import PipelineConfig
from scene_pred_pipeline.data_types import FlowResult, InstancePair
from scene_pred_pipeline.velocity_recovery import VelocityRecovery

if not torch.cuda.is_available():
    pytest.skip("CUDA required (local recovery backend)", allow_module_level=True)

DEV = torch.device("cuda:0")


def test_recovery_never_crosses_global_track_ids():
    config = PipelineConfig()
    recovery = VelocityRecovery(config)

    current = torch.tensor(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=torch.float32, device=DEV
    )
    ids = torch.tensor([1, 2], dtype=torch.int64, device=DEV)
    pair = InstancePair(
        previous_stamp_ns=0,
        current_stamp_ns=500_000_000,
        common_track_ids=(1, 2),
        previous_points=current,
        current_points=current,
        previous_track_ids=ids,
        current_track_ids=ids,
        dt_s=0.5,
    )
    zeros = torch.zeros((2, 3), dtype=torch.float32, device=DEV)
    flow_result = FlowResult(
        # Both anchors are deliberately colocated; only ID separation can keep
        # their opposite velocities from being mixed.
        source_anchors=zeros,
        warped_anchors=zeros,
        anchor_flow=torch.tensor(
            [[0.5, 0.0, 0.0], [-0.5, 0.0, 0.0]], dtype=torch.float32, device=DEV
        ),
        source_anchor_track_ids=ids,
        target_input_keep_mask=torch.ones(2, dtype=torch.bool, device=DEV),
    )

    # Displacement 0.5 m over dt 0.5 s -> +-1 m/s per track.
    velocity = recovery.recover(pair, flow_result).cpu()
    torch.testing.assert_close(velocity[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(velocity[1], torch.tensor([-1.0, 0.0, 0.0]))
