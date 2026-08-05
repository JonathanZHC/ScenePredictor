from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .config import PipelineConfig
from .data_types import (
    MotionState,
    Track,
    TrackObservation,
    TrackedFrame,
    VoxelizedFrame,
    VoxelizedObject,
)
from .voxel import (
    voxel_iou,
    voxel_neighbor_coverage,
)


def _aabb_gap_numpy(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    separation = np.maximum(
        np.maximum(
            min_a - max_b,
            min_b - max_a,
        ),
        0.0,
    )
    return float(np.linalg.norm(separation))


class SimpleObjectTracker:
    """Current-geometry tracking and multi-lag motion classification.

    Identity association never extrapolates position or differentiates a noisy
    visible-point centroid into a predicted velocity.
    """

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config
        self.next_track_id = 0
        self.previous_objects: list[
            VoxelizedObject
        ] = []
        self.tracks: dict[int, Track] = {}

    def _new_track(
        self,
        item: VoxelizedObject,
    ) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = Track(
            track_id=track_id,
            last_object=item,
        )
        return track_id

    @staticmethod
    def _geometry_table(
        objects: list[VoxelizedObject],
    ) -> np.ndarray:
        if not objects:
            return np.empty(
                (0, 9),
                dtype=np.float32,
            )
        return torch.stack(
            [
                torch.cat(
                    (
                        item.centroid_world,
                        item.aabb_min_world,
                        item.aabb_max_world,
                    )
                )
                for item in objects
            ]
        ).detach().float().cpu().numpy()

    def _assign_tracks(
        self,
        current: list[VoxelizedObject],
    ) -> None:
        current_geometry = self._geometry_table(
            current
        )
        previous_geometry = self._geometry_table(
            self.previous_objects
        )

        pending: list[
            tuple[
                int,
                int,
                float,
                float,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for current_index, item in enumerate(current):
            for previous_index, previous in enumerate(
                self.previous_objects
            ):
                if item.class_id != previous.class_id:
                    continue

                centroid_distance = float(
                    np.linalg.norm(
                        current_geometry[current_index, :3]
                        - previous_geometry[previous_index, :3]
                    )
                )
                box_gap = _aabb_gap_numpy(
                    current_geometry[current_index, 3:6],
                    current_geometry[current_index, 6:9],
                    previous_geometry[previous_index, 3:6],
                    previous_geometry[previous_index, 6:9],
                )
                if not (
                    centroid_distance
                    <= self.config.tracking.centroid_gate_m
                    or box_gap
                    <= self.config.tracking.aabb_gap_gate_m
                ):
                    continue

                current_in_previous = (
                    voxel_neighbor_coverage(
                        item.voxel_keys,
                        previous.voxel_coords,
                        self.config.tracking.voxel_neighbor_radius,
                    )
                )
                previous_in_current = (
                    voxel_neighbor_coverage(
                        previous.voxel_keys,
                        item.voxel_coords,
                        self.config.tracking.voxel_neighbor_radius,
                    )
                )
                pending.append(
                    (
                        current_index,
                        previous_index,
                        centroid_distance,
                        box_gap,
                        current_in_previous,
                        previous_in_current,
                    )
                )

        candidates: list[
            tuple[float, int, int]
        ] = []
        if pending:
            coverages = torch.stack(
                [
                    torch.stack(
                        (item[4], item[5])
                    )
                    for item in pending
                ]
            ).detach().float().cpu().numpy()

            for item, values in zip(
                pending,
                coverages,
                strict=True,
            ):
                (
                    current_index,
                    previous_index,
                    centroid_distance,
                    box_gap,
                    *_unused,
                ) = item
                current_in_previous = float(values[0])
                previous_in_current = float(values[1])
                coverage_max = max(
                    current_in_previous,
                    previous_in_current,
                )
                coverage_mean = 0.5 * (
                    current_in_previous
                    + previous_in_current
                )
                if (
                    coverage_max
                    < self.config.tracking.voxel_coverage_threshold
                ):
                    continue

                normalized_gap = min(
                    centroid_distance
                    / max(
                        self.config.tracking.centroid_gate_m,
                        1.0e-6,
                    ),
                    box_gap
                    / max(
                        self.config.tracking.aabb_gap_gate_m,
                        1.0e-6,
                    ),
                    1.0,
                )
                spatial_score = 1.0 - normalized_gap

                # max coverage handles partial occlusion; mean coverage rewards
                # mutually consistent visible surfaces.
                voxel_score = 0.5 * (
                    coverage_max + coverage_mean
                )
                score = (
                    self.config.tracking.voxel_weight
                    * voxel_score
                    + self.config.tracking.spatial_weight
                    * spatial_score
                )
                candidates.append(
                    (
                        score,
                        current_index,
                        previous_index,
                    )
                )

        used_current: set[int] = set()
        used_previous: set[int] = set()
        for _, current_index, previous_index in sorted(
            candidates,
            reverse=True,
        ):
            if (
                current_index in used_current
                or previous_index in used_previous
            ):
                continue
            current[current_index].track_id = (
                self.previous_objects[
                    previous_index
                ].track_id
            )
            used_current.add(current_index)
            used_previous.add(previous_index)

        for item in current:
            if item.track_id < 0:
                item.track_id = self._new_track(item)

    def _classify_motion(
        self,
        item: VoxelizedObject,
        frame_index: int,
        stamp_ns: int,
    ) -> MotionState:
        track = self.tracks[item.track_id]
        speeds: list[torch.Tensor] = []
        ious: list[torch.Tensor] = []

        for lag in self.config.tracking.history_lags:
            old = track.history.get(
                frame_index - lag
            )
            if old is None:
                continue

            dt_s = (
                stamp_ns - old.stamp_ns
            ) * 1.0e-9
            if dt_s <= 0.0:
                continue

            displacement = torch.linalg.norm(
                item.centroid_world
                - old.centroid_world
            )
            speeds.append(displacement / dt_s)
            ious.append(
                voxel_iou(
                    item.voxel_keys,
                    old.voxel_keys,
                )
            )

        if (
            len(speeds)
            < self.config.tracking.minimum_history_matches
        ):
            return MotionState.STATIC

        median_speed = torch.stack(
            speeds
        ).median()
        median_iou = torch.stack(
            ious
        ).median()

        moving = (
            median_speed
            > self.config.tracking.centroid_speed_threshold_mps
        ) | (
            median_iou
            < self.config.tracking.voxel_iou_threshold
        )
        return (
            MotionState.MOVING
            if bool(moving)
            else MotionState.STATIC
        )

    def _moving_masks(
        self,
        frame: VoxelizedFrame,
    ) -> dict[str, torch.Tensor]:
        if not (
            self.config.runtime.enable_visualization
            and self.config.output.publish_moving_masks
        ):
            return {}

        moving_members: dict[
            str,
            list[torch.Tensor],
        ] = defaultdict(list)
        for item in frame.objects:
            if (
                item.motion_state
                != MotionState.MOVING
            ):
                continue
            for member in item.members:
                moving_members[
                    member.camera_name
                ].append(member.mask_original)

        output: dict[str, torch.Tensor] = {}
        for camera_name, result in (
            frame.camera_results.items()
        ):
            masks = moving_members.get(
                camera_name,
                [],
            )
            if masks:
                output[camera_name] = torch.stack(
                    masks
                ).any(dim=0)
            else:
                output[camera_name] = torch.zeros(
                    result.camera.depth.shape,
                    device=result.camera.depth.device,
                    dtype=torch.bool,
                )
        return output

    def update(
        self,
        frame: VoxelizedFrame,
    ) -> TrackedFrame:
        self._assign_tracks(frame.objects)

        max_lag = max(
            self.config.tracking.history_lags,
            default=0,
        )
        for item in frame.objects:
            item.motion_state = self._classify_motion(
                item,
                frame.frame_index,
                frame.stamp_ns,
            )
            track = self.tracks[item.track_id]
            track.last_object = item
            track.history[
                frame.frame_index
            ] = TrackObservation(
                frame_index=frame.frame_index,
                stamp_ns=frame.stamp_ns,
                centroid_world=item.centroid_world,
                voxel_keys=item.voxel_keys,
                points=item.points,
            )

            oldest = (
                frame.frame_index - max_lag - 1
            )
            for key in tuple(track.history):
                if key < oldest:
                    del track.history[key]

        moving_masks = self._moving_masks(frame)
        self.previous_objects = frame.objects

        return TrackedFrame(
            frame_index=frame.frame_index,
            stamp_ns=frame.stamp_ns,
            objects=frame.objects,
            background_points=frame.background_points,
            background_keys=frame.background_keys,
            moving_masks=moving_masks,
        )
