import torch

from scene_pred_pipeline.config import PipelineConfig
from scene_pred_pipeline.data_types import FlowInput, FlowResult
from scene_pred_pipeline.velocity_recovery import VelocityRecovery


def test_recovery_never_crosses_global_track_ids():
    config = PipelineConfig()
    recovery = VelocityRecovery(config)

    current = torch.tensor(
        [[0.01, 0.0, 0.0], [0.02, 0.0, 0.0]], dtype=torch.float32
    )
    flow_input = FlowInput(
        previous_candidates=current,
        current_candidates=current,
        previous_candidate_track_ids=torch.tensor([1, 2]),
        current_candidate_track_ids=torch.tensor([1, 2]),
        previous_anchors=torch.zeros((2, 3)),
        current_anchors=torch.zeros((2, 3)),
        previous_anchor_track_ids=torch.tensor([1, 2]),
        current_anchor_track_ids=torch.tensor([1, 2]),
        current_dense_points=current,
        current_dense_track_ids=torch.tensor([1, 2]),
        common_track_ids=(1, 2),
        dt_s=0.033,
    )
    flow_result = FlowResult(
        # Both anchors are deliberately colocated; only ID separation can keep
        # their opposite velocities from being mixed.
        source_anchors=torch.zeros((2, 3)),
        warped_anchors=torch.zeros((2, 3)),
        anchor_velocity=torch.tensor(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float32
        ),
    )

    velocity = recovery.recover(flow_input, flow_result)
    torch.testing.assert_close(velocity[0], torch.tensor([1.0, 0.0, 0.0]))
    torch.testing.assert_close(velocity[1], torch.tensor([-1.0, 0.0, 0.0]))
