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
class ModelConfig:
    yolo_weights: str = "yolo11n-seg.pt"
    yolo_image_size: int = 640
    yolo_confidence: float = 0.25
    yolo_iou: float = 0.70
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "openai"
    clip_crop_padding: float = 0.10
    clip_batch_size: int = 64


@dataclass(frozen=True)
class MaskConfig:
    object_erosion_pixels: int = 2
    background_dilation_pixels: int = 4


@dataclass(frozen=True)
class DepthConfig:
    min_m: float = 0.10
    max_m: float = 5.00


@dataclass(frozen=True)
class MultiViewConfig:
    centroid_gate_m: float = 0.30
    reprojection_points: int = 64
    minimum_visible_points: int = 16
    occlusion_tolerance_m: float = 0.03
    depth_tolerance_m: float = 0.05
    reprojection_threshold: float = 0.50
    clip_threshold: float = 0.65
    high_class_confidence: float = 0.75
    reprojection_weight: float = 0.65
    clip_weight: float = 0.25
    centroid_weight: float = 0.10


@dataclass(frozen=True)
class VoxelConfig:
    size_m: float = 0.01
    origin_world: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class TrackingConfig:
    centroid_gate_m: float = 0.15
    clip_threshold: float = 0.60
    position_weight: float = 0.65
    appearance_weight: float = 0.35
    high_class_confidence: float = 0.75
    history_lags: tuple[int, ...] = (1, 2, 4, 8)
    centroid_speed_threshold_mps: float = 0.03
    voxel_iou_threshold: float = 0.60


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
    publish_background: bool = True
    publish_static_objects: bool = True
    publish_moving_objects: bool = True
    publish_velocity_cloud: bool = True
    publish_velocity_markers: bool = True
    publish_moving_masks: bool = True
    publish_annotated_rgb: bool = True
    velocity_marker_stride: int = 32
    velocity_marker_scale: float = 0.20
    profile_interval_frames: int = 30


@dataclass(frozen=True)
class RuntimeConfig:
    device: str = "cuda:0"
    enable_cuda_timing: bool = True
    allow_tf32: bool = True


@dataclass(frozen=True)
class PipelineConfig:
    ros: RosConfig = field(default_factory=RosConfig)
    models: ModelConfig = field(default_factory=ModelConfig)
    masks: MaskConfig = field(default_factory=MaskConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    multiview: MultiViewConfig = field(default_factory=MultiViewConfig)
    voxel: VoxelConfig = field(default_factory=VoxelConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)
    flow: FlowConfig = field(default_factory=FlowConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)


def _construct(dataclass_type, values: dict[str, Any] | None):
    values = dict(values or {})
    tuple_fields = {
        "camera_names",
        "origin_world",
        "history_lags",
    }
    for key in tuple_fields:
        if key in values and isinstance(values[key], list):
            values[key] = tuple(values[key])
    return dataclass_type(**values)


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    return PipelineConfig(
        ros=_construct(RosConfig, raw.get("ros")),
        models=_construct(ModelConfig, raw.get("models")),
        masks=_construct(MaskConfig, raw.get("masks")),
        depth=_construct(DepthConfig, raw.get("depth")),
        multiview=_construct(MultiViewConfig, raw.get("multiview")),
        voxel=_construct(VoxelConfig, raw.get("voxel")),
        tracking=_construct(TrackingConfig, raw.get("tracking")),
        flow=_construct(FlowConfig, raw.get("flow")),
        recovery=_construct(RecoveryConfig, raw.get("recovery")),
        output=_construct(OutputConfig, raw.get("output")),
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
    )
