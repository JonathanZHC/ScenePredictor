from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Collection, Mapping

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

    @property
    def static_only(self) -> bool:
        """No tracked classes: the pipeline skips SAM3/EfficientTAM/DifFlow and the merged
        scene cloud is the dense static background with zero velocity (pure obstacle avoidance)."""
        return len(self.tracked_prompts) == 0

    @property
    def max_tracked_instances(self) -> int:
        """Upper bound on simultaneously tracked instances (sum of per-class capacities)."""
        return int(sum(capacity for _, capacity in self.tracked_prompts))


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
    # Dense-input shortcuts (frames with at least this many anchors; 0 = off):
    # level-1 selection by strided indices instead of FPS, and hierarchical
    # feature-space KNN at the fine level. Weights are untouched.
    fast_top_level_min_points: int = 0
    hier_cosine_min_points: int = 0
    # Pyramid level-1 size for inputs above 1024 anchors (a 1024-anchor input is
    # level 1 itself). Upstream DifFlow3D samples 2048; every point bucket other
    # than 1024 must be at least this large.
    level1_points: int = 1024


@dataclass(frozen=True)
class DifFlowSpatialScaleConfig:
    enable: bool = True
    target_model_volume: float = 2.0
    fixed_spatial_scale: float = 1.0


@dataclass(frozen=True)
class DifFlowCanonicalNormalizationConfig:
    """Piecewise-constant canonical normalisation (difflow3d.runtime.normalization).

    x_model = k (x_world - c) + model_center with k = model_radius / anchor RMS
    radius; one transform per pair, re-anchored only when the RMS radius drifts by
    more than scale_ratio or the centroid by more than center_shift anchor radii.
    Replaces auto_spatial_scale (enable at most one of them). Defaults are the
    FT3D_s train medians the canonical checkpoints were fine-tuned with.
    """

    enable: bool = False
    model_center: tuple[float, float, float] = (-0.52, -1.57, 21.36)
    model_radius: float = 8.97
    scale_ratio: float = 1.5
    center_shift: float = 0.3


@dataclass(frozen=True)
class DifFlowOutlierFilterConfig:
    enabled: bool = False
    # Expensive, intentionally opt-in debugging: rebuild voxel-2 components and
    # print every decision. RViz removed-point output has its own output flag.
    detailed_output: bool = False
    min_component_size_ratio: float = 0.05


@dataclass(frozen=True)
class DifFlowPreprocessingConfig:
    # Maximum anchor count (CUDA graphs are captured per bucket, see below).
    fps_points: int = 2048
    second_candidate_ratio: float = 1.1
    final_selection: str = "uniform"
    # Exact anchor counts to run at. A frame uses the largest bucket <= its
    # candidate count, so points are never duplicated unless the frame has
    # fewer than the smallest bucket (1024, the network minimum). None = only
    # fps_points (legacy repeat-padding behaviour).
    point_buckets: tuple[int, ...] | None = None
    # Morton-sort the anchors; required for model.fast_top_level_min_points.
    sort_anchors_morton: bool = False
    auto_spatial_scale: DifFlowSpatialScaleConfig = field(
        default_factory=DifFlowSpatialScaleConfig
    )
    canonical_normalization: DifFlowCanonicalNormalizationConfig = field(
        default_factory=DifFlowCanonicalNormalizationConfig
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


CALIBRATION_REGIONS = ("ellipsoid", "ball")
CALIBRATION_MODES = ("global", "mondrian")


@dataclass(frozen=True)
class DifFlowCalibrationTable:
    """Conformal scales of one region kind.

    ``global_scales`` maps level -> scale; ``mondrian_scales`` maps level -> one
    scale per bin of the predicted displacement |flow| in mm, bin i holding
    mondrian_edges_mm[i-1] <= |flow| < mondrian_edges_mm[i] (np.digitize).
    """

    global_scales: tuple[tuple[float, float], ...] = ()
    mondrian_edges_mm: tuple[float, ...] = ()
    mondrian_scales: tuple[tuple[float, tuple[float, ...]], ...] = ()


@dataclass(frozen=True)
class DifFlowUncertaintyCalibrationConfig:
    """Conformal calibration of the flow covariance head.

    Sigma is the covariance of an anchor's one-pair displacement. With the scale s
    at coverage ``level`` and d = true minus predicted displacement, d lies in the
    calibrated region with probability ``level``:

    * ``region: ellipsoid``: d^T Sigma^-1 d <= s^2 (the ellipsoid s * chol(Sigma));
    * ``region: ball``: |d| <= s * det(Sigma)^(1/6).

    ``mode: global`` uses one scale per level, ``mondrian`` one per bin of the
    predicted displacement. Each region kind has its own tables; they hold only
    for the backbone, head and preprocessing they were fitted with (``source``).
    """

    region: str = "ellipsoid"
    mode: str = "mondrian"
    level: float = 0.95
    # Free-text provenance of the tables (dataset, split, checkpoints, preprocessing).
    source: str = ""
    ellipsoid: DifFlowCalibrationTable = field(default_factory=DifFlowCalibrationTable)
    ball: DifFlowCalibrationTable = field(default_factory=DifFlowCalibrationTable)

    def table(self) -> DifFlowCalibrationTable:
        """Tables of the configured region."""
        if self.region not in CALIBRATION_REGIONS:
            raise ValueError(
                "difflow.uncertainty_head.calibration.region must be ellipsoid or ball, "
                f"got {self.region!r}"
            )
        return getattr(self, self.region)

    def global_scale(self) -> float:
        """Global scale of the configured region at ``level``."""
        return float(
            _level_entry(self.table().global_scales, self.level, name=f"{self.region} global")
        )

    def mondrian_bin_scales(self) -> tuple[float, ...]:
        """Per-bin scales of the configured region at ``level`` (len(edges_mm) + 1 entries)."""
        return tuple(
            _level_entry(self.table().mondrian_scales, self.level, name=f"{self.region} mondrian")
        )


@dataclass(frozen=True)
class DifFlowUncertaintyHeadConfig:
    """Optional per-anchor flow covariance head run inside the DifFlow CUDA graphs."""

    enable: bool = False
    # Resolved like model.checkpoint: relative to flow.repo_path when not absolute.
    checkpoint: str = ""
    calibration: DifFlowUncertaintyCalibrationConfig = field(
        default_factory=DifFlowUncertaintyCalibrationConfig
    )


@dataclass(frozen=True)
class DifFlowConfig:
    """Deployment settings loaded once from the standalone difflow.yaml."""

    runtime: DifFlowRuntimeConfig = field(default_factory=DifFlowRuntimeConfig)
    model: DifFlowModelConfig = field(default_factory=DifFlowModelConfig)
    preprocessing: DifFlowPreprocessingConfig = field(
        default_factory=DifFlowPreprocessingConfig
    )
    recovery: DifFlowRecoveryConfig = field(default_factory=DifFlowRecoveryConfig)
    uncertainty_head: DifFlowUncertaintyHeadConfig = field(
        default_factory=DifFlowUncertaintyHeadConfig
    )


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
    # Voxel size of the rest-scene lattice. None = reuse the tracker's
    # shared_voxel_grid.voxel_size_m (5 mm by default); a larger value downsamples
    # the cloud handed to the safety filter. Required in static-only mode.
    voxel_size_m: float | None = None
    # Depth validity range for the rest scene when no tracker config is available
    # (static-only mode); with a tracker its postprocess limits are used.
    depth_min_m: float = 0.15
    depth_max_m: float = 5.0
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
    # Flow-uncertainty regions at the velocity-line tips, on by default and drawn
    # whenever the DifFlow uncertainty_head runs (no topic without it): at most
    # uncertainty_marker_max, spread over the lines, at most uncertainty_marker_publish_hz
    # (on top of visualization_publish_hz; a rate >= visualization_publish_hz publishes
    # them with every visualization update, on the velocity-line frames).
    publish_uncertainty_markers: bool = True
    uncertainty_marker_max: int = 150
    uncertainty_marker_alpha: float = 0.35
    uncertainty_marker_publish_hz: float = 10.0
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
    # An explicitly empty tracked_prompts list selects the static-only mode (no
    # tracker, no DifFlow; see TrackerConfig.static_only). excluded_prompts are
    # meaningless without a tracker.
    if not tracked and excluded:
        raise ValueError(
            "tracker.excluded_prompts requires at least one tracked class "
            "(static-only mode has no tracker to apply exclusions)"
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


def _number(value: Any, *, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be a number, got {value!r}") from error
    if not math.isfinite(number):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return number


def _positive(value: Any, *, name: str) -> float:
    number = _number(value, name=name)
    if number <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}")
    return number


def _level(value: Any, *, name: str) -> float:
    level = _number(value, name=name)
    if not 0.0 < level < 1.0:
        raise ValueError(f"{name} must be a coverage level in (0, 1), got {value!r}")
    return level


def _level_entry(table: tuple[tuple[float, Any], ...], level: float, *, name: str) -> Any:
    for key, value in table:
        if math.isclose(key, level, rel_tol=0.0, abs_tol=1.0e-9):
            return value
    raise ValueError(
        f"difflow.uncertainty_head.calibration.level {level} is not a level of the "
        f"{name} table (levels: {[key for key, _ in table]})"
    )


def _level_table(value: Any, *, name: str, row_length: int | None) -> tuple[tuple[float, Any], ...]:
    """Parse ``{level: scale}`` (row_length None) or ``{level: [scale per bin]}``."""
    rows: list[tuple[float, Any]] = []
    for raw_level, raw_scale in _mapping(value, name=name).items():
        level = _level(raw_level, name=f"{name} level")
        if any(math.isclose(level, key, rel_tol=0.0, abs_tol=1.0e-9) for key, _ in rows):
            raise ValueError(f"{name} defines level {level} twice")
        if row_length is None:
            scale: Any = _positive(raw_scale, name=f"{name}[{level}]")
        else:
            if not isinstance(raw_scale, (list, tuple)) or len(raw_scale) != row_length:
                raise ValueError(
                    f"{name}[{level}] must list {row_length} scales "
                    "(one per displacement bin = len(edges_mm) + 1)"
                )
            scale = tuple(_positive(item, name=f"{name}[{level}]") for item in raw_scale)
        rows.append((level, scale))
    return tuple(sorted(rows, key=lambda row: row[0]))


def _no_extra_keys(keys: Collection[str], *, name: str) -> None:
    if keys:
        raise ValueError(f"Unknown keys in {name}: {sorted(keys)}")


def _known_keys(
    dataclass_type,
    values: dict[str, Any],
    *,
    name: str,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Reject keys of ``values`` that are no field of ``dataclass_type`` (minus ``exclude``)."""
    allowed = {item.name for item in fields(dataclass_type)} - set(exclude)
    _no_extra_keys([key for key in values if key not in allowed], name=name)
    return values


def _calibration_table(value: Any, *, name: str) -> DifFlowCalibrationTable:
    values = _mapping(value, name=name)

    global_values = _mapping(values.pop("global", None), name=f"{name}.global")
    global_scales = _level_table(
        global_values.pop("scales", None), name=f"{name}.global.scales", row_length=None
    )
    _no_extra_keys(global_values, name=f"{name}.global")

    mondrian_values = _mapping(values.pop("mondrian", None), name=f"{name}.mondrian")
    raw_edges = mondrian_values.pop("edges_mm", None) or ()
    if not isinstance(raw_edges, (list, tuple)):
        raise ValueError(f"{name}.mondrian.edges_mm must be a list of displacements in mm")
    edges = tuple(_number(edge, name=f"{name}.mondrian.edges_mm") for edge in raw_edges)
    if any(edge < 0.0 for edge in edges) or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"{name}.mondrian.edges_mm must be non-negative and strictly increasing")
    mondrian_scales = _level_table(
        mondrian_values.pop("scales", None),
        name=f"{name}.mondrian.scales",
        row_length=len(edges) + 1,
    )
    _no_extra_keys(mondrian_values, name=f"{name}.mondrian")
    _no_extra_keys(values, name=name)
    return DifFlowCalibrationTable(
        global_scales=global_scales,
        mondrian_edges_mm=edges,
        mondrian_scales=mondrian_scales,
    )


def _uncertainty_calibration_config(value: Any) -> DifFlowUncertaintyCalibrationConfig:
    base = "difflow.uncertainty_head.calibration"
    values = _mapping(value, name=base)

    tables = _mapping(values.pop("tables", None), name=f"{base}.tables")
    region_tables = {
        region: _calibration_table(tables.pop(region, None), name=f"{base}.tables.{region}")
        for region in CALIBRATION_REGIONS
    }
    _no_extra_keys(tables, name=f"{base}.tables")

    _known_keys(
        DifFlowUncertaintyCalibrationConfig, values, name=base, exclude=CALIBRATION_REGIONS
    )
    if "level" in values:
        values["level"] = _level(values["level"], name=f"{base}.level")
    if "source" in values:
        values["source"] = str(values["source"] or "").strip()
    calibration = DifFlowUncertaintyCalibrationConfig(**region_tables, **values)
    if calibration.region not in CALIBRATION_REGIONS:
        raise ValueError(f"{base}.region must be ellipsoid or ball, got {calibration.region!r}")
    if calibration.mode not in CALIBRATION_MODES:
        raise ValueError(f"{base}.mode must be global or mondrian, got {calibration.mode!r}")
    return calibration


def _uncertainty_head_config(value: Any) -> DifFlowUncertaintyHeadConfig:
    name = "difflow.uncertainty_head"
    values = _mapping(value, name=name)
    calibration = _uncertainty_calibration_config(values.pop("calibration", None))
    _known_keys(DifFlowUncertaintyHeadConfig, values, name=name, exclude=("calibration",))
    head = DifFlowUncertaintyHeadConfig(calibration=calibration, **values)
    if not isinstance(head.enable, bool):
        raise ValueError(f"{name}.enable must be boolean")
    if head.enable:
        if not str(head.checkpoint).strip():
            raise ValueError(f"{name}.checkpoint is required when enabled")
        # The level must exist in the table of the configured region and mode.
        if calibration.mode == "global":
            calibration.global_scale()
        else:
            calibration.mondrian_bin_scales()
    return head


def check_uncertainty_head_meta(meta: Mapping[str, Any], difflow: DifFlowConfig) -> None:
    """Reject a covariance head that does not fit the configured DifFlow backbone.

    * The head reads the finest-level GRU state of the last iteration, which
      depends on the coarse/middle/fine schedule: ``meta['iters']`` ("coarse,
      middle,fine") must equal difflow.model.iterations.
    * A head trained on canonically normalised inputs (``meta['canonical']``)
      needs difflow.preprocessing.canonical_normalization.
    """
    _check_head_iterations(meta, difflow.model.iterations)
    if meta.get("canonical", False) and not difflow.preprocessing.canonical_normalization.enable:
        raise ValueError(
            "DifFlow uncertainty head was trained on canonically normalised inputs "
            "(checkpoint meta canonical: True); enable "
            "difflow.preprocessing.canonical_normalization (and turn "
            "auto_spatial_scale off)"
        )


def _check_head_iterations(meta: Mapping[str, Any], iterations: DifFlowIterationConfig) -> None:
    raw = meta.get("iters")
    if raw is None:
        raise ValueError("DifFlow uncertainty head checkpoint does not record 'iters'")
    if isinstance(raw, bool):
        raise ValueError(f"Invalid uncertainty head 'iters' entry: {raw!r}")
    try:
        if isinstance(raw, int):
            head_iterations = (raw, raw, raw)
        else:
            parts = raw.split(",") if isinstance(raw, str) else list(raw)
            head_iterations = tuple(int(str(part).strip()) for part in parts)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid uncertainty head 'iters' entry: {raw!r}") from error
    expected = (iterations.coarse, iterations.middle, iterations.fine)
    if head_iterations != expected:
        raise ValueError(
            "DifFlow uncertainty head was trained with iterations "
            f"(coarse, middle, fine) = {head_iterations}, but difflow.model.iterations "
            f"is {expected}"
        )


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

    canonical_values = _known_keys(
        DifFlowCanonicalNormalizationConfig,
        _mapping(
            preprocessing_values.pop("canonical_normalization", None),
            name="difflow.preprocessing.canonical_normalization",
        ),
        name="difflow.preprocessing.canonical_normalization",
    )
    if "model_center" in canonical_values:
        center = canonical_values["model_center"]
        if not isinstance(center, (list, tuple)) or len(center) != 3:
            raise ValueError(
                "difflow.preprocessing.canonical_normalization.model_center "
                "must have 3 entries"
            )
        canonical_values["model_center"] = tuple(
            _number(
                value,
                name="difflow.preprocessing.canonical_normalization.model_center",
            )
            for value in center
        )
    canonical_normalization = _construct(
        DifFlowCanonicalNormalizationConfig, canonical_values
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
        canonical_normalization=canonical_normalization,
        outlier_filter=outlier_filter,
        **preprocessing_values,
    )
    recovery = _construct(
        DifFlowRecoveryConfig,
        _mapping(raw.get("recovery"), name="difflow.recovery"),
    )
    uncertainty_head = _uncertainty_head_config(raw.get("uncertainty_head"))

    if runtime.cuda_graph_warmup < 1:
        raise ValueError("difflow.runtime.cuda_graph_warmup must be >= 1")
    if preprocessing.fps_points < 1024:
        raise ValueError("difflow.preprocessing.fps_points must be >= 1024")
    if preprocessing.second_candidate_ratio <= 1.0:
        raise ValueError("difflow.preprocessing.second_candidate_ratio must be > 1")
    if preprocessing.final_selection not in {"fps", "uniform"}:
        raise ValueError("difflow.preprocessing.final_selection must be fps or uniform")
    if preprocessing.point_buckets is not None:
        buckets = [int(value) for value in preprocessing.point_buckets]
        if not buckets or min(buckets) < 1024 or max(buckets) > preprocessing.fps_points:
            raise ValueError(
                "difflow.preprocessing.point_buckets must lie in "
                "[1024, fps_points]"
            )
    level1_points = model.level1_points
    if (
        isinstance(level1_points, bool)
        or not isinstance(level1_points, int)
        or level1_points < 1024
    ):
        raise ValueError(
            f"difflow.model.level1_points must be an integer >= 1024, got {level1_points!r}"
        )
    # The runner's buckets default to (fps_points,); a 1024-anchor input skips level-1 sampling.
    buckets = [int(value) for value in (preprocessing.point_buckets or (preprocessing.fps_points,))]
    too_small = [bucket for bucket in buckets if bucket != 1024 and bucket < level1_points]
    if too_small:
        raise ValueError(
            f"difflow.preprocessing point buckets {too_small} are smaller than "
            f"difflow.model.level1_points={level1_points}; every bucket other than "
            "1024 must be at least level1_points"
        )
    if model.fast_top_level_min_points > 0 and not preprocessing.sort_anchors_morton:
        raise ValueError(
            "difflow.model.fast_top_level_min_points requires "
            "difflow.preprocessing.sort_anchors_morton: true"
        )
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
    canonical = preprocessing.canonical_normalization
    if not isinstance(canonical.enable, bool):
        raise ValueError(
            "difflow.preprocessing.canonical_normalization.enable must be boolean"
        )
    if canonical.enable and preprocessing.auto_spatial_scale.enable:
        raise ValueError(
            "difflow.preprocessing.canonical_normalization replaces auto_spatial_scale; "
            "set preprocessing.auto_spatial_scale.enable: false"
        )
    _positive(
        canonical.model_radius,
        name="difflow.preprocessing.canonical_normalization.model_radius",
    )
    if _number(
        canonical.scale_ratio,
        name="difflow.preprocessing.canonical_normalization.scale_ratio",
    ) <= 1.0:
        raise ValueError(
            "difflow.preprocessing.canonical_normalization.scale_ratio must be > 1"
        )
    _positive(
        canonical.center_shift,
        name="difflow.preprocessing.canonical_normalization.center_shift",
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
        uncertainty_head=uncertainty_head,
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
    if cfg.voxel_size_m is not None and not (math.isfinite(cfg.voxel_size_m) and cfg.voxel_size_m > 0.0):
        raise ValueError("scene_cloud.voxel_size_m must be a positive number or null")
    if not (0.0 <= cfg.depth_min_m < cfg.depth_max_m):
        raise ValueError("scene_cloud.depth_min_m/depth_max_m must satisfy 0 <= min < max")
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
    if not isinstance(output.publish_uncertainty_markers, bool):
        raise ValueError("output.publish_uncertainty_markers must be boolean")
    uncertainty_marker_max = output.uncertainty_marker_max
    if (
        isinstance(uncertainty_marker_max, bool)
        or not isinstance(uncertainty_marker_max, int)
        or uncertainty_marker_max < 1
    ):
        raise ValueError("output.uncertainty_marker_max must be an integer >= 1")
    uncertainty_marker_alpha = _number(
        output.uncertainty_marker_alpha, name="output.uncertainty_marker_alpha"
    )
    if not 0.0 < uncertainty_marker_alpha <= 1.0:
        raise ValueError(
            f"output.uncertainty_marker_alpha must be in (0, 1], got {uncertainty_marker_alpha!r}"
        )
    _positive(
        output.uncertainty_marker_publish_hz, name="output.uncertainty_marker_publish_hz"
    )

    tracker = _tracker_config(raw.get("tracker"), base_dir=base_dir)
    scene_cloud = _scene_cloud_config(raw.get("scene_cloud"))
    if tracker.static_only and scene_cloud.enabled and scene_cloud.voxel_size_m is None:
        raise ValueError(
            "static-only mode (empty tracker.tracked_prompts) needs scene_cloud.voxel_size_m, "
            "since no tracker lattice is available to reuse"
        )

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
