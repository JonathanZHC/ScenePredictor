from __future__ import annotations

import math
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
    camera_frame: str = "{camera}_color_optical_frame"
    queue_depth: int = 4
    sync_slop_seconds: float = 0.001
    multiview_sync_slop_seconds: float = 0.02
    tf_timeout_ms: float = 10.0


@dataclass(frozen=True)
class TrackerConfig:
    """ScenePredictor-side integration settings for MultiViewRGBDTracker."""

    config_path: str = "/workspace/configs/tracking.yaml"
    checkpoint_root: str = "/workspace/checkpoints"
    tracked_prompts: tuple[tuple[str, int], ...] = (("human", 1),)
    excluded_prompts: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class FlowConfig:
    """ScenePredictor integration settings for the always-on DifFlow stage."""

    # Use the ABI-matched, prebuilt copy from the container image. The source
    # submodule under /workspace is still used at image build time.
    repo_path: str = "/opt/DifFlow3D"
    config_path: str = "/workspace/configs/difflow.yaml"

    # ScenePredictor streaming policy rather than a DifFlow model hyperparameter.
    max_frame_gap_s: float = 0.20


@dataclass(frozen=True)
class RecoveryConfig:
    """ScenePredictor-only recovery semantics."""

    # Keep recovery instance-conditioned so nearby independently moving objects
    # cannot blend velocities. Numerical recovery settings live in difflow.yaml.
    restrict_same_track: bool = True


@dataclass(frozen=True)
class DifFlowRuntimeConfig:
    device: str = "cuda"
    enable_tf32: bool = True
    cuda_graph_warmup: int = 10
    validate_finite: bool = False


@dataclass(frozen=True)
class DifFlowIterationConfig:
    coarse: int = 4
    middle: int = 2
    fine: int = 2


@dataclass(frozen=True)
class DifFlowModelConfig:
    checkpoint: str = "checkpoints/model_difflow_355_0.0114.pth"
    iterations: DifFlowIterationConfig = field(default_factory=DifFlowIterationConfig)
    uncertainty: float = 0.20
    strict_checkpoint: bool = True
    disable_bn_running_stats: bool = True


@dataclass(frozen=True)
class DifFlowSpatialScaleConfig:
    enable: bool = True
    target_model_volume: float = 2.0
    fixed_spatial_scale: float = 1.0


@dataclass(frozen=True)
class DifFlowOutlierFilterConfig:
    enabled: bool = False
    # Expensive, intentionally opt-in debugging: rebuild voxel-2 components and
    # print every decision. RViz removed-point output has its own output flag.
    detailed_output: bool = False
    min_component_size_ratio: float = 0.05


@dataclass(frozen=True)
class DifFlowPreprocessingConfig:
    fps_points: int = 2048
    second_candidate_ratio: float = 1.1
    final_selection: str = "uniform"
    auto_spatial_scale: DifFlowSpatialScaleConfig = field(
        default_factory=DifFlowSpatialScaleConfig
    )
    outlier_filter: DifFlowOutlierFilterConfig = field(
        default_factory=DifFlowOutlierFilterConfig
    )


@dataclass(frozen=True)
class DifFlowRecoveryConfig:
    softmax_sigma_m: float = 0.025
    backend: str = "local"
    local_radius_sigma: float = 4.0
    local_hash_size_factor: float = 4.0
    report_local_stats: bool = False
    chunk_size: int = 4096


@dataclass(frozen=True)
class DifFlowConfig:
    """Deployment settings loaded once from the standalone difflow.yaml."""

    runtime: DifFlowRuntimeConfig = field(default_factory=DifFlowRuntimeConfig)
    model: DifFlowModelConfig = field(default_factory=DifFlowModelConfig)
    preprocessing: DifFlowPreprocessingConfig = field(
        default_factory=DifFlowPreprocessingConfig
    )
    recovery: DifFlowRecoveryConfig = field(default_factory=DifFlowRecoveryConfig)


@dataclass(frozen=True)
class SceneCloudConfig:
    """Merged full-scene cloud built on the GPU at the end of every cycle.

    rows [0, num_dynamic)  tracked-instance points with DifFlow/recovered velocity
    rows [num_dynamic, N)  rest-scene points (dense depth minus tracked/excluded
                           masks, voxel-deduplicated) with zero velocity, track id 0
    This is the product consumed in-process by the safety filter.
    """

    enabled: bool = True
    include_rest_points: bool = True
    # Optional axis-aligned crop in the world frame (meters). None = no crop.
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None
    # Throttled ROS copy (/scene_predictor/scene_cloud) for RViz / recording. 0 disables.
    publish_hz: float = 5.0


@dataclass(frozen=True)
class OutputConfig:
    publish_tracked_objects: bool = True
    publish_rest_scene: bool = True
    publish_velocity_cloud: bool = True
    publish_velocity_markers: bool = True
    publish_tracked_masks: bool = True
    publish_annotated_rgb: bool = True
    publish_flow_anchors: bool = True
    # Independent from DifFlow's expensive detailed per-block text output.
    publish_removed_outlier_points: bool = False
    # Maximum update rate for visualization-only ROS topics. 0 = unthrottled.
    visualization_publish_hz: float = 10.0
    velocity_marker_stride: int = 32
    # Marker endpoint = point + velocity * flow_dt_s * factor.
    velocity_marker_scale_factor: float = 2.0
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
    scene_cloud: SceneCloudConfig = field(default_factory=SceneCloudConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    # Loaded from flow.config_path; it is intentionally not duplicated inside
    # the ScenePredictor default.yaml.
    difflow: DifFlowConfig = field(default_factory=DifFlowConfig)


def _construct(dataclass_type, values: dict[str, Any] | None):
    values = dict(values or {})
    if "camera_names" in values and isinstance(values["camera_names"], list):
        values["camera_names"] = tuple(values["camera_names"])
    return dataclass_type(**values)


def _mapping(value: Any, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a YAML mapping")
    return dict(value)


def _resolve_from(base_dir: Path, value: str) -> str:
    path = Path(value).expanduser()
    return str(path.resolve() if path.is_absolute() else (base_dir / path).resolve())


def _prompt_specs(value: Any, *, name: str) -> tuple[tuple[str, int], ...]:
    """Parse ScenePredictor prompt entries as immutable ``(label, capacity)`` pairs."""
    if value is None:
        return ()
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{name} must be a YAML list of [label, capacity] entries")

    output: list[tuple[str, int]] = []
    seen: set[str] = set()
    for item in value:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            label = str(item[0]).strip()
            capacity = int(item[1])
        elif isinstance(item, dict):
            raw_label = item.get("name", item.get("text", item.get("prompt")))
            raw_capacity = item.get(
                "capacity", item.get("max_instances", item.get("count", 1))
            )
            if raw_label is None:
                raise ValueError(
                    f"{name} mapping entries need name/text/prompt and capacity"
                )
            label = str(raw_label).strip()
            capacity = int(raw_capacity)
        else:
            raise ValueError(
                f"Each {name} entry must be [label, capacity] or a mapping; got {item!r}"
            )

        if not label:
            raise ValueError(f"{name} contains an empty semantic label")
        if capacity <= 0:
            raise ValueError(
                f"{name} capacity for {label!r} must be > 0, got {capacity}"
            )
        if label in seen:
            raise ValueError(f"{name} contains duplicate semantic class {label!r}")
        seen.add(label)
        output.append((label, capacity))
    return tuple(output)


def _tracker_config(value: Any, *, base_dir: Path) -> TrackerConfig:
    values = _mapping(value, name="tracker")
    has_prompt_keys = "tracked_prompts" in values or "excluded_prompts" in values
    tracked = _prompt_specs(
        values.pop("tracked_prompts", None), name="tracker.tracked_prompts"
    )
    excluded = _prompt_specs(
        values.pop("excluded_prompts", None), name="tracker.excluded_prompts"
    )

    # Preserve the dataclass default only for genuinely old configs that specify
    # neither key. Once either key is present, the supplied lists are authoritative.
    if not has_prompt_keys:
        tracked = TrackerConfig().tracked_prompts

    overlap = {label for label, _ in tracked} & {label for label, _ in excluded}
    if overlap:
        raise ValueError(
            "The same semantic class cannot be both tracked and excluded: "
            + ", ".join(sorted(overlap))
        )
    if not tracked:
        raise ValueError(
            "tracker.tracked_prompts must contain at least one class; the current "
            "3-D alignment workspace is sized from tracked classes only"
        )

    if "config_path" in values:
        values["config_path"] = _resolve_from(base_dir, str(values["config_path"]))
    if "checkpoint_root" in values:
        values["checkpoint_root"] = _resolve_from(
            base_dir, str(values["checkpoint_root"])
        )
    return TrackerConfig(
        tracked_prompts=tracked,
        excluded_prompts=excluded,
        **values,
    )


def _iteration_config(value: Any) -> DifFlowIterationConfig:
    if isinstance(value, bool):
        raise ValueError("difflow model.iterations must be an integer or mapping")
    if isinstance(value, int):
        if value < 1:
            raise ValueError("difflow model.iterations must be >= 1")
        return DifFlowIterationConfig(value, value, value)

    values = _mapping(value, name="difflow model.iterations")
    required = ("coarse", "middle", "fine")
    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(
            "difflow model.iterations must define coarse/middle/fine; "
            f"missing {missing}"
        )
    iterations = DifFlowIterationConfig(
        coarse=int(values["coarse"]),
        middle=int(values["middle"]),
        fine=int(values["fine"]),
    )
    if min(iterations.coarse, iterations.middle, iterations.fine) < 1:
        raise ValueError("all DifFlow recurrent iteration counts must be >= 1")
    return iterations


def _load_difflow_config(path: Path) -> DifFlowConfig:
    with path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = _mapping(raw, name=f"DifFlow config {path}")

    runtime = _construct(
        DifFlowRuntimeConfig,
        _mapping(raw.get("runtime"), name="difflow.runtime"),
    )

    model_values = _mapping(raw.get("model"), name="difflow.model")
    model_iterations = _iteration_config(model_values.pop("iterations", 4))
    model = DifFlowModelConfig(
        iterations=model_iterations,
        **model_values,
    )

    preprocessing_values = _mapping(
        raw.get("preprocessing"), name="difflow.preprocessing"
    )
    spatial_scale_value = preprocessing_values.pop("auto_spatial_scale", None)
    if isinstance(spatial_scale_value, bool):
        # Keep older ScenePredictor deployments readable while the canonical
        # configuration uses the nested DifFlow3D structure.
        spatial_scale = DifFlowSpatialScaleConfig(
            enable=spatial_scale_value,
            target_model_volume=float(
                preprocessing_values.pop("target_model_volume", 2.0)
            ),
            fixed_spatial_scale=float(
                preprocessing_values.pop("fixed_spatial_scale", 1.0)
            ),
        )
    else:
        spatial_scale = _construct(
            DifFlowSpatialScaleConfig,
            _mapping(
                spatial_scale_value,
                name="difflow.preprocessing.auto_spatial_scale",
            ),
        )
        if "target_model_volume" in preprocessing_values or (
            "fixed_spatial_scale" in preprocessing_values
        ):
            raise ValueError(
                "Move target_model_volume and fixed_spatial_scale under "
                "difflow.preprocessing.auto_spatial_scale."
            )

    outlier_filter = _construct(
        DifFlowOutlierFilterConfig,
        _mapping(
            preprocessing_values.pop("outlier_filter", None),
            name="difflow.preprocessing.outlier_filter",
        ),
    )
    preprocessing = DifFlowPreprocessingConfig(
        auto_spatial_scale=spatial_scale,
        outlier_filter=outlier_filter,
        **preprocessing_values,
    )
    recovery = _construct(
        DifFlowRecoveryConfig,
        _mapping(raw.get("recovery"), name="difflow.recovery"),
    )

    if runtime.cuda_graph_warmup < 1:
        raise ValueError("difflow.runtime.cuda_graph_warmup must be >= 1")
    if preprocessing.fps_points < 1024:
        raise ValueError("difflow.preprocessing.fps_points must be >= 1024")
    if preprocessing.second_candidate_ratio <= 1.0:
        raise ValueError("difflow.preprocessing.second_candidate_ratio must be > 1")
    if preprocessing.final_selection not in {"fps", "uniform"}:
        raise ValueError("difflow.preprocessing.final_selection must be fps or uniform")
    if preprocessing.auto_spatial_scale.target_model_volume <= 0.0:
        raise ValueError(
            "difflow.preprocessing.auto_spatial_scale.target_model_volume "
            "must be positive"
        )
    if preprocessing.auto_spatial_scale.fixed_spatial_scale <= 0.0:
        raise ValueError(
            "difflow.preprocessing.auto_spatial_scale.fixed_spatial_scale "
            "must be positive"
        )
    if not isinstance(preprocessing.outlier_filter.detailed_output, bool):
        raise ValueError(
            "difflow.preprocessing.outlier_filter.detailed_output must be boolean"
        )
    ratio = preprocessing.outlier_filter.min_component_size_ratio
    if not 0.0 <= ratio < 1.0:
        raise ValueError(
            "difflow.preprocessing.outlier_filter."
            "min_component_size_ratio must be in [0, 1)"
        )
    if recovery.softmax_sigma_m <= 0.0:
        raise ValueError("difflow.recovery.softmax_sigma_m must be positive")

    return DifFlowConfig(
        runtime=runtime,
        model=model,
        preprocessing=preprocessing,
        recovery=recovery,
    )


def load_config(path: str | Path) -> PipelineConfig:
    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    raw = _mapping(raw, name=f"ScenePredictor config {config_path}")
    return load_config_from_mapping(raw, base_dir=config_path.parent)


def _scene_cloud_config(value: Any) -> SceneCloudConfig:
    values = dict(_mapping(value, name="scene_cloud"))
    for key in ("workspace_min", "workspace_max"):
        if values.get(key) is not None:
            bounds = tuple(float(v) for v in values[key])
            if len(bounds) != 3:
                raise ValueError(f"scene_cloud.{key} must have 3 entries")
            values[key] = bounds
    cfg = _construct(SceneCloudConfig, values)
    if (cfg.workspace_min is None) != (cfg.workspace_max is None):
        raise ValueError("scene_cloud.workspace_min and workspace_max must be set together")
    if cfg.publish_hz < 0.0 or not math.isfinite(cfg.publish_hz):
        raise ValueError("scene_cloud.publish_hz must be finite and >= 0")
    return cfg


def load_config_from_mapping(raw: dict[str, Any], *, base_dir: str | Path) -> PipelineConfig:
    """Build a PipelineConfig from an already-parsed mapping.

    Relative paths (tracker/flow config_path, repo_path) resolve against
    ``base_dir``. Used by load_config() and by embedding applications whose
    own yaml carries the ScenePredictor settings as one section.
    """
    raw = _mapping(raw, name="ScenePredictor config")
    base_dir = Path(base_dir)

    flow_values = _mapping(raw.get("flow"), name="flow")
    if "enabled" in flow_values:
        raise ValueError(
            "flow.enabled has been removed: DifFlow is always enabled in "
            "ScenePredictor. Remove the key from default.yaml."
        )
    if "input_voxel_size_m" in flow_values:
        raise ValueError(
            "flow.input_voxel_size_m has been removed. ScenePredictor now reads "
            "shared_voxel_grid.voxel_size_m directly from the upstream tracker "
            "configuration."
        )

    if "config_path" in flow_values:
        flow_values["config_path"] = _resolve_from(
            base_dir, str(flow_values["config_path"])
        )
    if "repo_path" in flow_values:
        flow_values["repo_path"] = _resolve_from(
            base_dir, str(flow_values["repo_path"])
        )
    flow = _construct(FlowConfig, flow_values)

    difflow_config_path = Path(flow.config_path).expanduser().resolve()
    difflow = _load_difflow_config(difflow_config_path)

    ros = _construct(RosConfig, raw.get("ros"))
    if not ros.camera_names:
        raise ValueError("ros.camera_names must contain at least one camera")
    if len(set(ros.camera_names)) != len(ros.camera_names):
        raise ValueError("ros.camera_names must not contain duplicates")
    if ros.queue_depth < 1:
        raise ValueError("ros.queue_depth must be >= 1")
    if ros.sync_slop_seconds < 0.0:
        raise ValueError("ros.sync_slop_seconds must be >= 0")
    if ros.multiview_sync_slop_seconds < 0.0:
        raise ValueError("ros.multiview_sync_slop_seconds must be >= 0")
    if ros.tf_timeout_ms < 0.0:
        raise ValueError("ros.tf_timeout_ms must be >= 0")

    output_values = _mapping(raw.get("output"), name="output")
    if "velocity_marker_scale" in output_values:
        raise ValueError(
            "output.velocity_marker_scale has been removed. Use "
            "output.velocity_marker_scale_factor instead; marker length is now "
            "velocity * actual_flow_dt * factor."
        )
    output = _construct(OutputConfig, output_values)
    visualization_publish_hz = float(output.visualization_publish_hz)
    if not math.isfinite(visualization_publish_hz) or visualization_publish_hz < 0.0:
        raise ValueError(
            "output.visualization_publish_hz must be finite and >= 0 "
            "(0 disables visualization throttling)"
        )
    velocity_marker_scale_factor = float(output.velocity_marker_scale_factor)
    if not math.isfinite(velocity_marker_scale_factor) or velocity_marker_scale_factor < 0.0:
        raise ValueError(
            "output.velocity_marker_scale_factor must be finite and >= 0"
        )

    tracker = _tracker_config(raw.get("tracker"), base_dir=base_dir)
    scene_cloud = _scene_cloud_config(raw.get("scene_cloud"))

    return PipelineConfig(
        ros=ros,
        tracker=tracker,
        flow=flow,
        recovery=_construct(RecoveryConfig, raw.get("recovery")),
        output=output,
        scene_cloud=scene_cloud,
        runtime=_construct(RuntimeConfig, raw.get("runtime")),
        difflow=difflow,
    )
