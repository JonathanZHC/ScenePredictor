"""Stand-ins for the DifFlow3D runner and the batched row forward (CPU), shared by the object-flow tests."""
from dataclasses import replace

import torch

from scene_pred_pipeline.config import PipelineConfig, RuntimeConfig
from scene_pred_pipeline.data_types import FlowResult, InstancePair

DT = 1 / 30


class FakePredictor:
    """Returns the current points as anchors with their ids and a flow of 1 m per axis; remembers what it was given."""

    outlier_filter_enabled = False
    detailed_outlier_output = False

    def __init__(self):
        self.calls = []
        self.resets = 0
        self.reject = None                       # world position of a current point to reject
        self._cached_target_stamp_ns = None
        self._cached_track_signature = None

    @staticmethod
    def calibration_scale(flow):
        return torch.full((flow.shape[0],), 2.0)

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


class FakeRowBatch:
    """The exact displacement between corresponding anchors, covariance 1e-6 I; remembers the launched row sets."""

    def __init__(self, anchors, row_counts):
        self.anchors, self.capacity = anchors, max(row_counts)
        self.launched = []
        self._pending = None

    def prepare(self):
        pass

    def launch(self, source, target):
        assert source.shape == target.shape and source.shape[1] == self.anchors and source.shape[0] <= self.capacity
        self.launched.append(source.shape[0])
        self._pending = target - source

    def collect(self):
        displacement, self._pending = self._pending, None
        return displacement, 1.0e-6 * torch.eye(3).expand(*displacement.shape[:2], 3, 3)


def config(instances: int = 8, **object_flow) -> PipelineConfig:
    """CPU pipeline config for ``instances`` trackable objects with ``difflow.object_flow`` fields replaced
    (``scheduled`` takes a dict)."""
    base = PipelineConfig(runtime=RuntimeConfig(device="cpu"))
    base = replace(base, tracker=replace(base.tracker, tracked_prompts=(("object", instances),)))
    section = base.difflow.object_flow
    if "scheduled" in object_flow:
        object_flow["scheduled"] = replace(section.scheduled, **object_flow["scheduled"])
    return replace(base, difflow=replace(base.difflow, object_flow=replace(section, **object_flow)))


def blob(center, n, seed, spread=0.05):
    generator = torch.Generator().manual_seed(seed)
    return torch.tensor(center) + spread * torch.rand(n, 3, generator=generator)


def pair(clouds, stamp=1, previous_stamp=0, previous=None):
    """clouds: {track id: [N, 3]} of the current frame; previous = current unless given. Stamps count frames."""
    previous = clouds if previous is None else previous
    ids = lambda frame: torch.cat([torch.full((cloud.shape[0],), track, dtype=torch.int32) for track, cloud in frame.items()])
    frame_ns = int(round(DT * 1e9))
    return InstancePair(previous_stamp_ns=previous_stamp * frame_ns, current_stamp_ns=stamp * frame_ns,
                        common_track_ids=tuple(clouds), previous_points=torch.cat(list(previous.values())),
                        current_points=torch.cat(list(clouds.values())), previous_track_ids=ids(previous),
                        current_track_ids=ids(clouds), dt_s=DT * (stamp - previous_stamp))
