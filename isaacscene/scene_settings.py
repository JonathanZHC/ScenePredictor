#!/usr/bin/env python3
"""Build tabletop-only static, dynamic and hybrid Isaac Sim scenes.

Every foreground object is a referenced Isaac Sim USD model and every object
stays on or above the tabletop. The table and ground are the only primitive
geometry retained from the original scene.

SimulationApp must be created before importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pxr import Gf, Sdf, UsdGeom, UsdLux, UsdShade

from isaac_asset_objects import (
    ASSET_CATALOG,
    AssetInstance,
    ObjectFootprint,
    create_asset_instance,
    footprint_for,
    set_asset_pose,
)


SceneMode = Literal[
    "static",
    "dynamic",
    "hybrid",
]
SCENE_MODES: tuple[str, str, str] = (
    "static",
    "dynamic",
    "hybrid",
)


@dataclass(frozen=True)
class SceneConfig:
    """Environment dimensions in metres."""

    table_top_center_z: float = 0.75
    table_top_size_x: float = 1.60
    table_top_size_y: float = 1.15
    table_top_thickness: float = 0.08
    ground_size: float = 5.0
    object_edge_margin_m: float = 0.018
    object_clearance_m: float = 0.018

    @property
    def table_surface_z(self) -> float:
        return (
            self.table_top_center_z
            + 0.5 * self.table_top_thickness
        )

    @property
    def table_half_extent_xy(
        self,
    ) -> tuple[float, float]:
        return (
            0.5 * self.table_top_size_x,
            0.5 * self.table_top_size_y,
        )


@dataclass(frozen=True)
class StaticPlacement:
    """One support-surface placement in its common usage frame."""

    key: str
    xy: tuple[float, float]
    yaw_deg: float = 0.0


@dataclass(frozen=True)
class SceneBuildResult:
    """Metadata required by the moving-object controller."""

    mode: SceneMode
    table_surface_z: float
    table_half_extent_xy: tuple[float, float]
    object_paths: dict[str, str]
    static_footprints: tuple[ObjectFootprint, ...]


DEFAULT_SCENE_CONFIG = SceneConfig()


# All eight classes appear in the static scene. The larger shelf occupies the
# rear-left tabletop; daily objects form a dense front row; the ball remains
# stationary in the rear-right area.
STATIC_LAYOUT: tuple[StaticPlacement, ...] = (
    StaticPlacement("shelf", (-0.30, 0.20), 0.0),
    StaticPlacement("ball", (0.48, 0.18), 0.0),
    StaticPlacement("cereal_box", (-0.56, -0.34), -10.0),
    StaticPlacement("food_can", (-0.30, -0.34), 0.0),
    StaticPlacement("bottle", (-0.08, -0.34), -6.0),
    StaticPlacement("mug", (0.16, -0.34), 24.0),
    StaticPlacement("banana", (0.40, -0.34), -16.0),
    StaticPlacement("pudding_box", (0.59, -0.10), 9.0),
)


# Hybrid mode keeps four daily objects stationary while shelf and ball are
# constructed and animated by MovingObjectController.
HYBRID_STATIC_LAYOUT: tuple[StaticPlacement, ...] = (
    StaticPlacement("cereal_box", (-0.54, -0.34), -10.0),
    StaticPlacement("mug", (-0.24, -0.34), 24.0),
    StaticPlacement("banana", (0.08, -0.34), -18.0),
    StaticPlacement("pudding_box", (0.34, -0.34), 8.0),
)


def _define_xform(
    stage,
    path: str,
) -> None:
    UsdGeom.Xform.Define(
        stage,
        Sdf.Path(path),
    )


def _set_xform(
    prim,
    translation: tuple[float, float, float],
    scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    rotation_xyz_deg: tuple[float, float, float] | None = None,
) -> None:
    xformable = UsdGeom.Xformable(prim)
    xformable.ClearXformOpOrder()
    xformable.AddTranslateOp().Set(
        Gf.Vec3d(*translation)
    )
    if rotation_xyz_deg is not None:
        xformable.AddRotateXYZOp().Set(
            Gf.Vec3f(*rotation_xyz_deg)
        )
    xformable.AddScaleOp().Set(
        Gf.Vec3f(*scale)
    )


def _create_material(
    stage,
    path: str,
    color: tuple[float, float, float],
    roughness: float,
) -> UsdShade.Material:
    """Create materials only for the shared table and ground."""

    material = UsdShade.Material.Define(
        stage,
        Sdf.Path(path),
    )
    shader = UsdShade.Shader.Define(
        stage,
        Sdf.Path(f"{path}/Shader"),
    )
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput(
        "diffuseColor",
        Sdf.ValueTypeNames.Color3f,
    ).Set(
        Gf.Vec3f(*color)
    )
    shader.CreateInput(
        "roughness",
        Sdf.ValueTypeNames.Float,
    ).Set(
        float(roughness)
    )
    material.CreateSurfaceOutput().ConnectToSource(
        shader.ConnectableAPI(),
        "surface",
    )
    return material


def _bind_material(
    prim,
    material: UsdShade.Material,
) -> None:
    UsdShade.MaterialBindingAPI.Apply(
        prim
    ).Bind(material)


def _create_environment_cube(
    stage,
    path: str,
    center: tuple[float, float, float],
    size_xyz: tuple[float, float, float],
    material: UsdShade.Material,
) -> None:
    cube = UsdGeom.Cube.Define(
        stage,
        Sdf.Path(path),
    )
    cube.CreateSizeAttr(1.0)
    _set_xform(
        cube.GetPrim(),
        center,
        size_xyz,
    )
    _bind_material(
        cube.GetPrim(),
        material,
    )


def _build_ground_and_table(
    stage,
    config: SceneConfig,
) -> None:
    ground_material = _create_material(
        stage,
        "/World/Materials/Ground",
        (0.18, 0.20, 0.23),
        0.85,
    )
    table_material = _create_material(
        stage,
        "/World/Materials/Table",
        (0.42, 0.20, 0.07),
        0.58,
    )

    _create_environment_cube(
        stage,
        "/World/Ground",
        (0.0, 0.0, -0.025),
        (
            config.ground_size,
            config.ground_size,
            0.05,
        ),
        ground_material,
    )
    _create_environment_cube(
        stage,
        "/World/Table/Top",
        (
            0.0,
            0.0,
            config.table_top_center_z,
        ),
        (
            config.table_top_size_x,
            config.table_top_size_y,
            config.table_top_thickness,
        ),
        table_material,
    )

    leg_height = (
        config.table_top_center_z
        - 0.5 * config.table_top_thickness
    )
    leg_z = 0.5 * leg_height
    leg_x = (
        0.5 * config.table_top_size_x
        - 0.10
    )
    leg_y = (
        0.5 * config.table_top_size_y
        - 0.10
    )
    for index, (x, y) in enumerate(
        (
            (-leg_x, -leg_y),
            (leg_x, -leg_y),
            (-leg_x, leg_y),
            (leg_x, leg_y),
        )
    ):
        _create_environment_cube(
            stage,
            f"/World/Table/Leg_{index}",
            (
                x,
                y,
                leg_z,
            ),
            (
                0.08,
                0.08,
                leg_height,
            ),
            table_material,
        )


def _build_static_assets(
    stage,
    *,
    config: SceneConfig,
    placements: tuple[StaticPlacement, ...],
) -> tuple[
    dict[str, str],
    tuple[ObjectFootprint, ...],
]:
    object_paths: dict[str, str] = {}
    footprints: list[ObjectFootprint] = []

    for index, placement in enumerate(placements):
        instance = create_asset_instance(
            stage,
            spec=ASSET_CATALOG[placement.key],
            root_path=(
                f"/World/StaticObjects/"
                f"Object_{index:02d}_{placement.key}"
            ),
        )
        set_asset_pose(
            instance,
            position_world=(
                placement.xy[0],
                placement.xy[1],
                config.table_surface_z,
            ),
            rotation_xyz_deg=(
                0.0,
                0.0,
                placement.yaw_deg,
            ),
        )
        object_paths[
            f"{placement.key}_{index}"
        ] = instance.root_path
        footprints.append(
            footprint_for(
                instance,
                placement.xy,
            )
        )

    _validate_static_layout(
        footprints,
        config,
    )
    return (
        object_paths,
        tuple(footprints),
    )


def _validate_static_layout(
    footprints: list[ObjectFootprint],
    config: SceneConfig,
) -> None:
    half_x, half_y = config.table_half_extent_xy

    for footprint in footprints:
        x, y = footprint.center_xy
        radius = footprint.radius_m
        if (
            abs(x)
            + radius
            + config.object_edge_margin_m
            > half_x
            or abs(y)
            + radius
            + config.object_edge_margin_m
            > half_y
        ):
            raise RuntimeError(
                "Static asset does not fit on the tabletop:\n"
                f"  object={footprint.name}\n"
                f"  xy={footprint.center_xy}\n"
                f"  radius={radius:.4f}\n"
                f"  table_half_extent={(half_x, half_y)}"
            )

    for first_index, first in enumerate(footprints):
        for second in footprints[first_index + 1 :]:
            dx = (
                first.center_xy[0]
                - second.center_xy[0]
            )
            dy = (
                first.center_xy[1]
                - second.center_xy[1]
            )
            distance = (
                dx * dx
                + dy * dy
            ) ** 0.5
            required = (
                first.radius_m
                + second.radius_m
                + config.object_clearance_m
            )
            if distance < required:
                raise RuntimeError(
                    "Static tabletop assets intersect according to "
                    "conservative measured footprints:\n"
                    f"  first={first.name} at {first.center_xy}\n"
                    f"  second={second.name} at {second.center_xy}\n"
                    f"  distance={distance:.4f}\n"
                    f"  required={required:.4f}"
                )


def _build_lighting(stage) -> None:
    dome = UsdLux.DomeLight.Define(
        stage,
        Sdf.Path("/World/Lights/Dome"),
    )
    dome.CreateIntensityAttr(500.0)
    dome.CreateColorAttr(
        Gf.Vec3f(0.95, 0.97, 1.0)
    )

    key = UsdLux.DistantLight.Define(
        stage,
        Sdf.Path("/World/Lights/Key"),
    )
    key.CreateIntensityAttr(2600.0)
    key.CreateAngleAttr(1.0)
    _set_xform(
        key.GetPrim(),
        (0.0, 0.0, 3.5),
        rotation_xyz_deg=(
            -38.0,
            20.0,
            24.0,
        ),
    )

    fill = UsdLux.SphereLight.Define(
        stage,
        Sdf.Path("/World/Lights/Fill"),
    )
    fill.CreateIntensityAttr(18000.0)
    fill.CreateRadiusAttr(0.35)
    fill.CreateColorAttr(
        Gf.Vec3f(1.0, 0.80, 0.68)
    )
    _set_xform(
        fill.GetPrim(),
        (-1.4, -1.0, 2.3),
    )


def build_scene(
    stage,
    scene_mode: SceneMode = "static",
    config: SceneConfig = DEFAULT_SCENE_CONFIG,
) -> SceneBuildResult:
    """Build one complete tabletop scene."""

    if scene_mode not in SCENE_MODES:
        raise ValueError(
            f"Unknown scene mode {scene_mode!r}; "
            f"expected one of {SCENE_MODES}."
        )

    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.z,
    )
    UsdGeom.SetStageMetersPerUnit(
        stage,
        1.0,
    )

    for path in (
        "/World",
        "/World/Table",
        "/World/StaticObjects",
        "/World/MovingObjects",
        "/World/Cameras",
        "/World/Lights",
        "/World/Materials",
    ):
        _define_xform(
            stage,
            path,
        )

    _build_ground_and_table(
        stage,
        config,
    )

    object_paths: dict[str, str] = {
        "table": "/World/Table",
    }
    static_footprints: tuple[
        ObjectFootprint,
        ...
    ] = ()

    if scene_mode == "static":
        (
            static_paths,
            static_footprints,
        ) = _build_static_assets(
            stage,
            config=config,
            placements=STATIC_LAYOUT,
        )
        object_paths.update(
            static_paths
        )
    elif scene_mode == "hybrid":
        (
            static_paths,
            static_footprints,
        ) = _build_static_assets(
            stage,
            config=config,
            placements=HYBRID_STATIC_LAYOUT,
        )
        object_paths.update(
            static_paths
        )

    _build_lighting(stage)

    return SceneBuildResult(
        mode=scene_mode,
        table_surface_z=config.table_surface_z,
        table_half_extent_xy=config.table_half_extent_xy,
        object_paths=object_paths,
        static_footprints=static_footprints,
    )
