#!/usr/bin/env python3
"""Animate non-intersecting tabletop Isaac Sim asset models.

Dynamic mode:
- a relatively large shelf moves slowly along the rear tabletop;
- a ball repeatedly bounces while translating parallel to the tabletop;
- a bottle, food can and mug move in separated front-row lanes.

Hybrid mode:
- the shelf and ball move;
- daily objects from scene_settings.py stay stationary.

All non-ball objects retain their common authored usage frame. In particular,
the shelf, bottle, food can and mug remain vertical and receive no roll/pitch.

SimulationApp must be created before importing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable

from isaac_asset_objects import (
    ASSET_CATALOG,
    AssetInstance,
    ObjectFootprint,
    create_asset_instance,
    set_asset_pose,
)


@dataclass(frozen=True)
class MotionPose:
    """One world pose at a trajectory time."""

    position_world: tuple[float, float, float]
    rotation_xyz_deg: tuple[float, float, float]


Trajectory = Callable[
    [float, float],
    MotionPose,
]


@dataclass(frozen=True)
class MotionDefinition:
    name: str
    asset_key: str
    trajectory: Trajectory


@dataclass
class _MovingRuntime:
    definition: MotionDefinition
    instance: AssetInstance


def _phase(
    elapsed_s: float,
    period_s: float,
) -> float:
    return (
        2.0
        * math.pi
        * elapsed_s
        / period_s
    )


def _shelf_motion(
    t: float,
    surface_z: float,
) -> MotionPose:
    """Slow installed-frame shelf motion along table X."""

    phase = _phase(
        t,
        10.0,
    )
    return MotionPose(
        position_world=(
            0.10 * math.sin(phase),
            0.22,
            surface_z,
        ),
        rotation_xyz_deg=(
            0.0,
            0.0,
            3.0 * math.sin(0.5 * phase),
        ),
    )


def _ball_motion(
    t: float,
    surface_z: float,
) -> MotionPose:
    """Parabolic bounce plus smooth motion parallel to the table.

    The root of the normalized ball is its lowest point. A zero lift therefore
    means contact with the tabletop. The repeated parabola has an instantaneous
    velocity reversal at contact, representing an idealized bounce impulse.
    """

    bounce_period_s = 1.20
    bounce_fraction = (
        t
        / bounce_period_s
    ) % 1.0
    bounce_height_m = (
        4.0
        * 0.22
        * bounce_fraction
        * (1.0 - bounce_fraction)
    )

    horizontal_phase = _phase(
        t,
        6.0,
    )
    x = (
        0.49
        + 0.055 * math.sin(horizontal_phase)
    )
    y = (
        -0.10
        + 0.012 * math.sin(
            0.5 * horizontal_phase
        )
    )

    return MotionPose(
        position_world=(
            x,
            y,
            surface_z + bounce_height_m,
        ),
        rotation_xyz_deg=(
            0.0,
            0.0,
            math.degrees(horizontal_phase) % 360.0,
        ),
    )


def _bottle_motion(
    t: float,
    surface_z: float,
) -> MotionPose:
    phase = _phase(
        t,
        8.4,
    )
    return MotionPose(
        position_world=(
            -0.53
            + 0.070 * math.sin(phase),
            -0.34,
            surface_z,
        ),
        rotation_xyz_deg=(
            0.0,
            0.0,
            8.0 * math.sin(0.7 * phase),
        ),
    )


def _food_can_motion(
    t: float,
    surface_z: float,
) -> MotionPose:
    phase = _phase(
        t,
        7.6,
    )
    return MotionPose(
        position_world=(
            -0.22
            + 0.060 * math.sin(phase),
            -0.34,
            surface_z,
        ),
        rotation_xyz_deg=(
            0.0,
            0.0,
            math.degrees(0.8 * phase) % 360.0,
        ),
    )


def _mug_motion(
    t: float,
    surface_z: float,
) -> MotionPose:
    phase = _phase(
        t,
        9.1,
    )
    return MotionPose(
        position_world=(
            0.10
            + 0.055 * math.sin(phase),
            -0.34,
            surface_z,
        ),
        rotation_xyz_deg=(
            0.0,
            0.0,
            25.0
            + 10.0 * math.sin(0.8 * phase),
        ),
    )


DYNAMIC_MOTIONS: tuple[
    MotionDefinition,
    ...
] = (
    MotionDefinition(
        name="moving_shelf",
        asset_key="shelf",
        trajectory=_shelf_motion,
    ),
    MotionDefinition(
        name="bouncing_ball",
        asset_key="ball",
        trajectory=_ball_motion,
    ),
    MotionDefinition(
        name="moving_bottle",
        asset_key="bottle",
        trajectory=_bottle_motion,
    ),
    MotionDefinition(
        name="moving_food_can",
        asset_key="food_can",
        trajectory=_food_can_motion,
    ),
    MotionDefinition(
        name="moving_mug",
        asset_key="mug",
        trajectory=_mug_motion,
    ),
)


HYBRID_MOTIONS: tuple[
    MotionDefinition,
    ...
] = (
    MotionDefinition(
        name="moving_shelf",
        asset_key="shelf",
        trajectory=_shelf_motion,
    ),
    MotionDefinition(
        name="bouncing_ball",
        asset_key="ball",
        trajectory=_ball_motion,
    ),
)


class MovingObjectController:
    """Create, validate and update moving tabletop asset models."""

    CLEARANCE_M = 0.020
    TABLE_EDGE_MARGIN_M = 0.018
    VALIDATION_DURATION_S = 60.0
    VALIDATION_SAMPLES = 1200
    SUPPORT_TOLERANCE_M = 1.0e-6
    UPRIGHT_TOLERANCE_DEG = 1.0e-6

    def __init__(
        self,
        stage,
        *,
        scene_mode: str,
        table_surface_z: float,
        table_half_extent_xy: tuple[float, float],
        static_footprints: tuple[
            ObjectFootprint,
            ...
        ] = (),
        speed_scale: float = 1.0,
    ) -> None:
        if scene_mode not in (
            "dynamic",
            "hybrid",
        ):
            raise ValueError(
                "MovingObjectController requires "
                "'dynamic' or 'hybrid' mode."
            )
        if speed_scale <= 0.0:
            raise ValueError(
                "speed_scale must be positive."
            )

        self.scene_mode = scene_mode
        self.table_surface_z = float(
            table_surface_z
        )
        self.table_half_extent_xy = (
            float(table_half_extent_xy[0]),
            float(table_half_extent_xy[1]),
        )
        self.static_footprints = tuple(
            static_footprints
        )
        self.speed_scale = float(
            speed_scale
        )

        definitions = (
            DYNAMIC_MOTIONS
            if scene_mode == "dynamic"
            else HYBRID_MOTIONS
        )

        self.objects: dict[
            str,
            _MovingRuntime,
        ] = {}

        for index, definition in enumerate(definitions):
            instance = create_asset_instance(
                stage,
                spec=ASSET_CATALOG[
                    definition.asset_key
                ],
                root_path=(
                    f"/World/MovingObjects/"
                    f"Object_{index:02d}_"
                    f"{definition.name}"
                ),
            )
            self.objects[
                definition.name
            ] = _MovingRuntime(
                definition=definition,
                instance=instance,
            )

        self._validate_all_trajectories()
        self.update(0.0)

    def _pose(
        self,
        runtime: _MovingRuntime,
        elapsed_s: float,
    ) -> MotionPose:
        return runtime.definition.trajectory(
            elapsed_s,
            self.table_surface_z,
        )

    def _validate_common_state(
        self,
        runtime: _MovingRuntime,
        pose: MotionPose,
        elapsed_s: float,
    ) -> None:
        instance = runtime.instance
        roll, pitch, _ = pose.rotation_xyz_deg
        root_z = float(
            pose.position_world[2]
        )

        if instance.pose_style != "ball":
            if (
                abs(float(roll))
                > self.UPRIGHT_TOLERANCE_DEG
                or abs(float(pitch))
                > self.UPRIGHT_TOLERANCE_DEG
            ):
                raise RuntimeError(
                    "Non-ball moving object left its common "
                    "authored frame:\n"
                    f"  object={runtime.definition.name}\n"
                    f"  pose_style={instance.pose_style}\n"
                    f"  t={elapsed_s:.4f}s\n"
                    f"  rotation={pose.rotation_xyz_deg}"
                )
            if (
                abs(
                    root_z
                    - self.table_surface_z
                )
                > self.SUPPORT_TOLERANCE_M
            ):
                raise RuntimeError(
                    "Supported moving object is not resting on "
                    "the tabletop:\n"
                    f"  object={runtime.definition.name}\n"
                    f"  t={elapsed_s:.4f}s\n"
                    f"  z={root_z:.6f}\n"
                    f"  surface={self.table_surface_z:.6f}"
                )
        else:
            if (
                root_z
                < self.table_surface_z
                - self.SUPPORT_TOLERANCE_M
            ):
                raise RuntimeError(
                    "Ball penetrates the tabletop:\n"
                    f"  t={elapsed_s:.4f}s\n"
                    f"  z={root_z:.6f}\n"
                    f"  surface={self.table_surface_z:.6f}"
                )

    def _validate_on_table(
        self,
        runtime: _MovingRuntime,
        pose: MotionPose,
        elapsed_s: float,
    ) -> None:
        x = float(
            pose.position_world[0]
        )
        y = float(
            pose.position_world[1]
        )
        radius = float(
            runtime.instance.footprint_radius_m
        )
        half_x, half_y = self.table_half_extent_xy

        if (
            abs(x)
            + radius
            + self.TABLE_EDGE_MARGIN_M
            > half_x
            or abs(y)
            + radius
            + self.TABLE_EDGE_MARGIN_M
            > half_y
        ):
            raise RuntimeError(
                "Moving asset leaves the tabletop footprint:\n"
                f"  object={runtime.definition.name}\n"
                f"  t={elapsed_s:.4f}s\n"
                f"  xy={(x, y)}\n"
                f"  radius={radius:.4f}\n"
                f"  table_half_extent={(half_x, half_y)}"
            )

    def _validate_against_static(
        self,
        runtime: _MovingRuntime,
        pose: MotionPose,
        elapsed_s: float,
    ) -> None:
        x = float(
            pose.position_world[0]
        )
        y = float(
            pose.position_world[1]
        )
        radius = float(
            runtime.instance.footprint_radius_m
        )

        for static in self.static_footprints:
            distance = math.hypot(
                x - static.center_xy[0],
                y - static.center_xy[1],
            )
            required = (
                radius
                + static.radius_m
                + self.CLEARANCE_M
            )
            if distance < required:
                raise RuntimeError(
                    "Moving trajectory intersects a static asset:\n"
                    f"  moving={runtime.definition.name}\n"
                    f"  static={static.name}\n"
                    f"  t={elapsed_s:.4f}s\n"
                    f"  distance={distance:.4f}\n"
                    f"  required={required:.4f}"
                )

    def _validate_pair(
        self,
        first_runtime: _MovingRuntime,
        first_pose: MotionPose,
        second_runtime: _MovingRuntime,
        second_pose: MotionPose,
        elapsed_s: float,
    ) -> None:
        distance_xy = math.hypot(
            (
                first_pose.position_world[0]
                - second_pose.position_world[0]
            ),
            (
                first_pose.position_world[1]
                - second_pose.position_world[1]
            ),
        )
        required = (
            first_runtime.instance.footprint_radius_m
            + second_runtime.instance.footprint_radius_m
            + self.CLEARANCE_M
        )

        # This conservative XY test remains active even when the ball is in the
        # air, so no moving object can pass underneath it or the shelf.
        if distance_xy < required:
            raise RuntimeError(
                "Moving trajectories intersect according to "
                "measured conservative footprints:\n"
                f"  first={first_runtime.definition.name}\n"
                f"  second={second_runtime.definition.name}\n"
                f"  t={elapsed_s:.4f}s\n"
                f"  distance_xy={distance_xy:.4f}\n"
                f"  required={required:.4f}"
            )

    def _validate_all_trajectories(
        self,
    ) -> None:
        runtimes = tuple(
            self.objects.values()
        )

        for sample_index in range(
            self.VALIDATION_SAMPLES
        ):
            alpha = (
                sample_index
                / max(
                    1,
                    self.VALIDATION_SAMPLES - 1,
                )
            )
            elapsed_s = (
                alpha
                * self.VALIDATION_DURATION_S
            )
            sampled = [
                (
                    runtime,
                    self._pose(
                        runtime,
                        elapsed_s,
                    ),
                )
                for runtime in runtimes
            ]

            for runtime, pose in sampled:
                self._validate_common_state(
                    runtime,
                    pose,
                    elapsed_s,
                )
                self._validate_on_table(
                    runtime,
                    pose,
                    elapsed_s,
                )
                self._validate_against_static(
                    runtime,
                    pose,
                    elapsed_s,
                )

            for first_index, (
                first_runtime,
                first_pose,
            ) in enumerate(sampled):
                for (
                    second_runtime,
                    second_pose,
                ) in sampled[
                    first_index + 1 :
                ]:
                    self._validate_pair(
                        first_runtime,
                        first_pose,
                        second_runtime,
                        second_pose,
                        elapsed_s,
                    )

        print(
            "[MOTION VALIDATION] "
            f"mode={self.scene_mode}, "
            f"samples={self.VALIDATION_SAMPLES}, "
            f"duration={self.VALIDATION_DURATION_S:.1f}s: passed",
            flush=True,
        )

    def update(
        self,
        elapsed_s: float,
    ) -> None:
        t = max(
            0.0,
            float(elapsed_s),
        ) * self.speed_scale

        for runtime in self.objects.values():
            pose = self._pose(
                runtime,
                t,
            )
            set_asset_pose(
                runtime.instance,
                position_world=pose.position_world,
                rotation_xyz_deg=pose.rotation_xyz_deg,
            )

    def description(self) -> str:
        return (
            ", ".join(
                self.objects.keys()
            )
            or "none"
        )
