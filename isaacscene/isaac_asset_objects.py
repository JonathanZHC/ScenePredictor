#!/usr/bin/env python3
"""Resolve, load, normalize and pose Isaac Sim foreground assets.

Foreground objects are always referenced USD models. Their original meshes,
hierarchy and materials are retained. Only the common ground/table geometry in
scene_settings.py uses simple primitives.

SimulationApp must be created before importing this module.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math
import os
import re
from typing import Any, Literal

import omni.client
from isaacsim.storage.native import get_assets_root_path
from pxr import Gf, Sdf, Usd, UsdGeom


PoseStyle = Literal[
    "upright",
    "horizontal",
    "installed",
    "ball",
]


@dataclass(frozen=True)
class AssetSearch:
    """Bounded asset-directory search used when no stable exact path exists."""

    relative_roots: tuple[str, ...]
    include_keywords: tuple[str, ...]
    reject_keywords: tuple[str, ...] = ()
    max_depth: int = 4
    max_entries: int = 3000


@dataclass(frozen=True)
class AssetSpec:
    """One semantic object and its available Isaac Sim asset locations."""

    key: str
    label: str
    candidate_relative_usd_paths: tuple[str, ...]
    pose_style: PoseStyle
    fixed_scale: float = 1.0
    target_longest_xy_m: float | None = None
    target_max_height_m: float | None = None
    search: AssetSearch | None = None
    override_environment_variable: str | None = None


@dataclass
class AssetInstance:
    """Runtime transform handles and measured normalized dimensions."""

    key: str
    label: str
    pose_style: PoseStyle
    root_path: str
    asset_url: str
    translate_op: Any
    rotate_op: Any
    size_x_m: float
    size_y_m: float
    height_m: float
    footprint_radius_m: float


@dataclass(frozen=True)
class ObjectFootprint:
    """Conservative circular XY footprint for deterministic clearance tests."""

    name: str
    center_xy: tuple[float, float]
    radius_m: float


ASSET_CATALOG: dict[str, AssetSpec] = {
    "cereal_box": AssetSpec(
        key="cereal_box",
        label="cereal box",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/003_cracker_box.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/003_cracker_box.usd",
        ),
        pose_style="upright",
        fixed_scale=0.90,
    ),
    "food_can": AssetSpec(
        key="food_can",
        label="food can",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/005_tomato_soup_can.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/005_tomato_soup_can.usd",
        ),
        pose_style="upright",
    ),
    "bottle": AssetSpec(
        key="bottle",
        label="bottle",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/006_mustard_bottle.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
        ),
        pose_style="upright",
        fixed_scale=0.92,
    ),
    "pudding_box": AssetSpec(
        key="pudding_box",
        label="pudding box",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/008_pudding_box.usd",
        ),
        pose_style="upright",
        fixed_scale=0.95,
    ),
    "banana": AssetSpec(
        key="banana",
        label="banana",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/011_banana.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/011_banana.usd",
        ),
        pose_style="horizontal",
    ),
    "mug": AssetSpec(
        key="mug",
        label="mug",
        candidate_relative_usd_paths=(
            "/Isaac/Props/YCB/Axis_Aligned/025_mug.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/025_mug.usd",
        ),
        pose_style="upright",
        fixed_scale=0.92,
    ),
    "ball": AssetSpec(
        key="ball",
        label="ball",
        candidate_relative_usd_paths=(
            # Prebuilt Isaac Sim USD asset. This is referenced as an asset;
            # the scene code does not construct a Sphere primitive.
            "/Isaac/Props/Shapes/sphere.usd",
            "/Isaac/Props/Shapes/Sphere.usd",
            "/Isaac/Props/YCB/Axis_Aligned/055_baseball.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/055_baseball.usd",
            "/Isaac/Props/YCB/Axis_Aligned/054_softball.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/054_softball.usd",
            "/Isaac/Props/YCB/Axis_Aligned/056_tennis_ball.usd",
            "/Isaac/Props/YCB/Axis_Aligned_Physics/056_tennis_ball.usd",
        ),
        pose_style="ball",
        target_longest_xy_m=0.12,
        target_max_height_m=0.12,
        search=AssetSearch(
            relative_roots=(
                "/Isaac/Props/YCB/Axis_Aligned",
                "/Isaac/Props/YCB/Axis_Aligned_Physics",
            ),
            include_keywords=("ball",),
            reject_keywords=(
                "material",
                "texture",
                "racquet",
            ),
            max_depth=1,
            max_entries=500,
        ),
        override_environment_variable="SCENEPREDICTOR_BALL_USD",
    ),
    "shelf": AssetSpec(
        key="shelf",
        label="shelf",
        # Exact names have changed across asset-pack releases. These are tried
        # first; a bounded folder search is used if none exists.
        candidate_relative_usd_paths=(
            "/Isaac/Environments/Simple_Warehouse/Props/SM_Rack_01.usd",
            "/Isaac/Environments/Simple_Warehouse/Props/SM_Rack_02.usd",
            "/Isaac/Environments/Simple_Warehouse/Props/SM_Rack_03.usd",
            "/Isaac/Environments/Modular_Warehouse/Props/SM_Rack_01.usd",
            "/Isaac/Environments/Modular_Warehouse/Props/SM_Rack_02.usd",
            "/Isaac/Environments/Modular_Warehouse/Props/Rack.usd",
            "/Isaac/Environments/Modular_Warehouse/Props/WarehouseRack.usd",
        ),
        pose_style="installed",
        target_longest_xy_m=0.44,
        target_max_height_m=0.42,
        search=AssetSearch(
            relative_roots=(
                "/Isaac/Environments/Modular_Warehouse/Props",
                "/Isaac/Environments/Simple_Warehouse",
            ),
            include_keywords=(
                "shelf",
                "rack",
            ),
            reject_keywords=(
                "material",
                "texture",
                "look",
                "collision",
                "proxy",
                "frame",
                "board",
                "rackshelf",
                "rack_shelf",
                "shelfboard",
                "shelf_board",
                "plank",
                "beam",
                "brace",
                "multiple",
                "full_warehouse",
                "with_forklift",
            ),
            max_depth=4,
            max_entries=3000,
        ),
        override_environment_variable="SCENEPREDICTOR_SHELF_USD",
    ),
}


_ASSETS_ROOT: str | None = None
_RESOLVED_URL_CACHE: dict[str, str] = {}


def get_isaac_assets_root() -> str:
    """Return and cache the configured Isaac Sim assets root URL."""

    global _ASSETS_ROOT
    if _ASSETS_ROOT is None:
        value = get_assets_root_path()
        if not value:
            raise RuntimeError(
                "Isaac Sim assets root is unavailable. Configure the default "
                "cloud asset root or mount the Isaac Sim assets pack."
            )
        _ASSETS_ROOT = str(value).rstrip("/")
    return _ASSETS_ROOT


def _join_url(parent: str, child: str) -> str:
    return parent.rstrip("/") + "/" + child.lstrip("/")


def _entry_relative_path(entry: Any) -> str:
    value = getattr(entry, "relative_path", None)
    if value is None:
        value = getattr(entry, "relativePath", None)
    return str(value or "")


def _entry_can_have_children(entry: Any) -> bool:
    try:
        return bool(
            entry.flags
            & omni.client.ItemFlags.CAN_HAVE_CHILDREN
        )
    except Exception:
        # Providers do not always expose complete flags. A later list call is
        # permitted to determine whether an extensionless item is a folder.
        return False


def _asset_exists(url: str) -> bool:
    result, _ = omni.client.stat(url)
    return result == omni.client.Result.OK


def _normalised_name(url: str) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        url.lower(),
    )


def _score_discovered_asset(
    url: str,
    search: AssetSearch,
) -> int | None:
    name = _normalised_name(url)
    if not url.lower().endswith((".usd", ".usda", ".usdc")):
        return None
    if any(keyword in name for keyword in search.reject_keywords):
        return None
    if not any(keyword in name for keyword in search.include_keywords):
        return None

    score = 0
    basename = _normalised_name(url.rsplit("/", maxsplit=1)[-1])

    # Prefer complete rack/shelving assemblies. RackShelf assets in the Simple
    # Warehouse pack are individual horizontal shelf boards and are rejected
    # before this point.
    if "rack" in basename:
        score += 140
    if "shelf" in basename:
        score += 80
    if "shelving" in basename:
        score += 100
    if "unit" in basename:
        score += 35
    if "assembly" in basename:
        score += 35
    if "set" in basename:
        score += 15
    if "warehouse" in name:
        score += 10

    # Prefer concise object-level filenames over deeply nested components.
    score -= url.count("/")
    score -= max(0, basename.count("_") - 3) * 3
    return score


def _discover_asset_urls(
    search: AssetSearch,
) -> list[str]:
    """Perform a bounded breadth-first directory scan."""

    assets_root = get_isaac_assets_root()
    queue: deque[tuple[str, int]] = deque(
        (
            assets_root + relative_root,
            0,
        )
        for relative_root in search.relative_roots
    )
    visited: set[str] = set()
    candidates: list[tuple[int, str]] = []
    examined = 0

    while queue and examined < search.max_entries:
        directory_url, depth = queue.popleft()
        if directory_url in visited:
            continue
        visited.add(directory_url)

        result, entries = omni.client.list(directory_url)
        if result != omni.client.Result.OK:
            continue

        for entry in entries:
            examined += 1
            relative_path = _entry_relative_path(entry)
            if not relative_path:
                continue
            child_url = _join_url(
                directory_url,
                relative_path,
            )
            score = _score_discovered_asset(
                child_url,
                search,
            )
            if score is not None:
                candidates.append((score, child_url))

            if depth >= search.max_depth:
                continue

            extension = os.path.splitext(relative_path)[1].lower()
            if (
                _entry_can_have_children(entry)
                or extension == ""
            ):
                queue.append(
                    (
                        child_url,
                        depth + 1,
                    )
                )

            if examined >= search.max_entries:
                break

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
        )
    )
    return [url for _, url in candidates]


def _candidate_urls(spec: AssetSpec) -> list[str]:
    urls: list[str] = []

    if spec.override_environment_variable:
        override = os.environ.get(
            spec.override_environment_variable,
            "",
        ).strip()
        if override:
            if "://" not in override and not override.startswith("/"):
                raise ValueError(
                    f"{spec.override_environment_variable} must be an "
                    f"absolute path or URL, got {override!r}."
                )
            if override.startswith("/Isaac/"):
                override = get_isaac_assets_root() + override
            elif override.startswith("/"):
                override = "file://" + override
            urls.append(override)

    assets_root = get_isaac_assets_root()
    urls.extend(
        assets_root + relative_path
        for relative_path in spec.candidate_relative_usd_paths
    )

    if spec.search is not None:
        urls.extend(
            _discover_asset_urls(spec.search)
        )

    # Stable de-duplication.
    return list(dict.fromkeys(urls))


def resolve_asset_url(spec: AssetSpec) -> str:
    """Resolve one object model and cache the selected URL."""

    cached = _RESOLVED_URL_CACHE.get(spec.key)
    if cached is not None:
        return cached

    attempted: list[str] = []
    for url in _candidate_urls(spec):
        attempted.append(url)
        if _asset_exists(url):
            _RESOLVED_URL_CACHE[spec.key] = url
            print(
                f"[ASSET] {spec.key}: {url}",
                flush=True,
            )
            return url

    attempted_text = "\n".join(
        f"  - {url}"
        for url in attempted
    )
    override_help = ""
    if spec.override_environment_variable:
        override_help = (
            "\nYou can explicitly select a compatible model with:\n"
            f"  {spec.override_environment_variable}=<absolute USD URL>"
        )
    raise FileNotFoundError(
        f"No Isaac Sim USD asset could be resolved for {spec.key!r}.\n"
        f"Attempted/discovered URLs:\n{attempted_text}"
        f"{override_help}"
    )


def _define_xform(stage, path: str):
    """Define an Xform after binding-independent path validation."""

    if not isinstance(path, str) or not path.startswith("/"):
        raise ValueError(
            f"USD prim path must be absolute, got {path!r}."
        )

    identifier = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
    components = [
        component
        for component in path.split("/")
        if component
    ]
    if (
        not components
        or any(
            identifier.fullmatch(component) is None
            for component in components
        )
    ):
        raise ValueError(
            "Invalid USD prim path. Every component must begin with a letter "
            "or underscore and contain only letters, digits or underscores: "
            f"{path!r}"
        )
    return UsdGeom.Xform.Define(
        stage,
        Sdf.Path(path),
    )


def _make_world_ops(prim) -> tuple[Any, Any]:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble,
        "world",
    )
    rotate_op = xformable.AddRotateXYZOp(
        UsdGeom.XformOp.PrecisionFloat,
        "world",
    )
    return translate_op, rotate_op


def _make_normalization_ops(prim) -> tuple[Any, Any, Any]:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(
        UsdGeom.XformOp.PrecisionDouble,
        "normalize",
    )
    rotate_op = xformable.AddRotateXYZOp(
        UsdGeom.XformOp.PrecisionFloat,
        "normalize",
    )
    scale_op = xformable.AddScaleOp(
        UsdGeom.XformOp.PrecisionFloat,
        "normalize",
    )
    return translate_op, rotate_op, scale_op


def _aligned_world_range(prim) -> Gf.Range3d:
    """Return a finite ordered bound without version-specific empty methods."""

    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
        ],
        useExtentsHint=True,
    )
    value = cache.ComputeWorldBound(prim).ComputeAlignedRange()
    minimum = value.GetMin()
    maximum = value.GetMax()

    numbers = (
        float(minimum[0]),
        float(minimum[1]),
        float(minimum[2]),
        float(maximum[0]),
        float(maximum[1]),
        float(maximum[2]),
    )
    finite = all(
        math.isfinite(number)
        for number in numbers
    )
    ordered = all(
        float(maximum[index]) >= float(minimum[index])
        for index in range(3)
    )
    if not finite or not ordered:
        raise RuntimeError(
            "Referenced asset has an invalid bound: "
            f"{prim.GetPath()}, min={tuple(minimum)}, max={tuple(maximum)}"
        )
    return value


def _range_size(
    value: Gf.Range3d,
) -> tuple[float, float, float]:
    minimum = value.GetMin()
    maximum = value.GetMax()
    return (
        float(maximum[0] - minimum[0]),
        float(maximum[1] - minimum[1]),
        float(maximum[2] - minimum[2]),
    )


def _required_uniform_scale(
    spec: AssetSpec,
    raw_size: tuple[float, float, float],
) -> float:
    raw_x, raw_y, raw_z = raw_size
    if min(raw_size) <= 0.0:
        raise RuntimeError(
            f"Asset {spec.key!r} has non-positive raw size {raw_size}."
        )

    scale = float(spec.fixed_scale)
    if spec.target_longest_xy_m is not None:
        scale = min(
            scale,
            float(spec.target_longest_xy_m)
            / max(raw_x, raw_y),
        )
    if spec.target_max_height_m is not None:
        scale = min(
            scale,
            float(spec.target_max_height_m)
            / raw_z,
        )
    if not math.isfinite(scale) or scale <= 0.0:
        raise RuntimeError(
            f"Invalid scale {scale} for asset {spec.key!r}."
        )
    return scale


def create_asset_instance(
    stage,
    *,
    spec: AssetSpec,
    root_path: str,
) -> AssetInstance:
    """Reference, uniformly scale and bottom-centre one authored model."""

    asset_url = resolve_asset_url(spec)

    root = _define_xform(
        stage,
        root_path,
    )
    root_translate, root_rotate = _make_world_ops(
        root.GetPrim()
    )
    root_translate.Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )
    root_rotate.Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )

    normalization = _define_xform(
        stage,
        f"{root_path}/Normalization",
    )
    (
        normalization_translate,
        normalization_rotate,
        normalization_scale,
    ) = _make_normalization_ops(
        normalization.GetPrim()
    )
    normalization_translate.Set(
        Gf.Vec3d(0.0, 0.0, 0.0)
    )
    normalization_rotate.Set(
        Gf.Vec3f(0.0, 0.0, 0.0)
    )
    normalization_scale.Set(
        Gf.Vec3f(1.0, 1.0, 1.0)
    )

    reference_path = (
        f"{root_path}/Normalization/Asset"
    )
    reference_prim = stage.DefinePrim(
        Sdf.Path(reference_path),
        "Xform",
    )
    reference_prim.GetReferences().AddReference(
        asset_url
    )
    stage.Load(reference_prim.GetPath())

    raw_range = _aligned_world_range(
        normalization.GetPrim()
    )
    raw_size = _range_size(raw_range)
    scale = _required_uniform_scale(
        spec,
        raw_size,
    )
    normalization_scale.Set(
        Gf.Vec3f(scale, scale, scale)
    )

    scaled_range = _aligned_world_range(
        normalization.GetPrim()
    )
    minimum = scaled_range.GetMin()
    maximum = scaled_range.GetMax()
    center_x = 0.5 * (
        float(minimum[0])
        + float(maximum[0])
    )
    center_y = 0.5 * (
        float(minimum[1])
        + float(maximum[1])
    )
    normalization_translate.Set(
        Gf.Vec3d(
            -center_x,
            -center_y,
            -float(minimum[2]),
        )
    )

    normalized_range = _aligned_world_range(
        normalization.GetPrim()
    )
    size_x, size_y, size_z = _range_size(
        normalized_range
    )
    footprint_radius = 0.5 * math.hypot(
        size_x,
        size_y,
    )

    print(
        f"[ASSET SIZE] {spec.key}: "
        f"{size_x:.3f} x {size_y:.3f} x {size_z:.3f} m, "
        f"scale={scale:.6f}, pose={spec.pose_style}",
        flush=True,
    )

    return AssetInstance(
        key=spec.key,
        label=spec.label,
        pose_style=spec.pose_style,
        root_path=root_path,
        asset_url=asset_url,
        translate_op=root_translate,
        rotate_op=root_rotate,
        size_x_m=size_x,
        size_y_m=size_y,
        height_m=size_z,
        footprint_radius_m=footprint_radius,
    )


def set_asset_pose(
    instance: AssetInstance,
    *,
    position_world: tuple[float, float, float],
    rotation_xyz_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Set the root pose while enforcing each model's common usage frame."""

    roll, pitch, yaw = (
        float(rotation_xyz_deg[0]),
        float(rotation_xyz_deg[1]),
        float(rotation_xyz_deg[2]),
    )

    if instance.pose_style != "ball":
        if abs(roll) > 1.0e-6 or abs(pitch) > 1.0e-6:
            raise ValueError(
                f"{instance.key!r} must remain in its authored "
                f"{instance.pose_style!r} frame; runtime roll/pitch is "
                f"not allowed. Received {(roll, pitch, yaw)}."
            )

    instance.translate_op.Set(
        Gf.Vec3d(
            float(position_world[0]),
            float(position_world[1]),
            float(position_world[2]),
        )
    )
    instance.rotate_op.Set(
        Gf.Vec3f(
            roll,
            pitch,
            yaw,
        )
    )


def footprint_for(
    instance: AssetInstance,
    center_xy: tuple[float, float],
) -> ObjectFootprint:
    return ObjectFootprint(
        name=instance.key,
        center_xy=(
            float(center_xy[0]),
            float(center_xy[1]),
        ),
        radius_m=float(instance.footprint_radius_m),
    )
