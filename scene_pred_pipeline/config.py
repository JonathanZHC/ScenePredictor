from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RosConfig:
    world_frame: str = "world"
    camera_names: tuple[str, ...] = ("camera_0", "camera_1")
    color_topic: str = "/{camera}/color/image_raw"
    depth_topic: str = "/{camera}/depth/image_raw"
    camera_info_topic: str = "/{camera}/camera_info"
    pose_topic: str = "/{camera}/pose"
    queue_depth: int = 4
    incomplete_timeout_ms: float = 50.0


@dataclass(frozen=True)
class TrackerConfig:
    """ScenePredictor-side integration settings for MultiViewRGBDTracker.

    Detector prompts, tracker capacities, postprocessing, cross-view fusion and
    cross-frame alignment remain owned by MultiViewRGBDTracker's tracking.yaml.
    """

    config_path: str = "/workspace/MultiViewRGBDTracker/configs/tracking.yaml"
    checkpoint_root: str = "/workspace/checkpoints"
    disable_internal_visualization: bool = True


@dataclass(frozen=True)
class FlowConfig:
    enabled: bool = True
    repo_path: str = "/opt/DifFlow3D"
    model_module: str = "model_difflow"
    checkpoint: str = (
        "/opt/DifFlow3D/pretrain_weights/model_difflow_355_0.0114.pth"
    )
    target_points: int = 2048
    pre_fps_factor: float = 2.0
    iterations: int = 4
    uncertainty: float = 0.20
    cuda_graph_warmup: int = 10
    enable_tf32: bool = True
    strict_checkpoint: bool = False
    max_frame_gap_s: float = 0.20


@dataclass(frozen=True)
class RecoveryConfig:
    knn: int = 3
    temperature_m: float = 0.02
    chunk_size: int = 8192
    restrict_same_track: bool = True


@dataclass(frozen=True)
class OutputConfig:
    publish_tracked_objects: bool = True
    publish_velocity_cloud: bool = True
    publish_velocity_markers: bool = True
    publish_tracked_masks: bool = True
    publish_annotated_rgb: bool = True
    publish_flow_anchors: bool = True
    velocity_marker_stride: int = 32
    velocity_marker_scale: float = 0.20
    profile_interval_frames: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    # ScenePredictor tensors may use cuda:0; tracker_adapter normalizes GPU 0
    # to the literal "cuda" required by the current SAM3 image builder.
    device: str = "cuda:0"
    enable_cuda_timing: bool = True
    allow_tf32: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    ros: RosConfig = field(default_factory=RosConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _construct(dataclass_type, values: dict[str, Any] | None):
    values = dict(values or {})
    if "camera_names" in values and isinstance(values["camera_names"], list):
        values["camera_names"] = tuple(values["camera_names"])
    return dataclass_type(**values)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    return PipelineConfig(
        ros=_construct(RosConfig, raw.get("ros")),
        tracker=_construct(TrackerConfig, raw.get("tracker")),
        flow=_construct(FlowConfig, raw.get("flow")),
        recovery=_construct(RecoveryConfig, raw.get("recovery")),
        output=_construct(OutputConfig, raw.get("output")),
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
    )
