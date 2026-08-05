from __future__ import annotations

from collections import defaultdict
import statistics

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
from .voxel import voxel_iou


class SimpleObjectTracker:
    """Step 4: previous-frame matching and median multi-lag motion decision."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.next_track_id = 0
        self.previous_objects: list[VoxelizedObject] = []
        self.tracks: dict[int, Track] = {}

    def _new_track(self, item: VoxelizedObject) -> int:
        track_id = self.next_track_id
        self.next_track_id += 1
        self.tracks[track_id] = Track(
            track_id=track_id,
            last_object=item,
        )
        return track_id

    def _assign_tracks(self, current: list[VoxelizedObject]) -> None:
        candidates: list[tuple[float, int, int]] = []

        for current_index, item in enumerate(current):
            for previous_index, previous in enumerate(self.previous_objects):
                distance = torch.linalg.norm(
                    item.centroid_world - previous.centroid_world
                )
                if float(distance) > self.config.tracking.centroid_gate_m:
                    continue
                if (
                    item.class_id != previous.class_id
                    and item.class_confidence
                    > self.config.tracking.high_class_confidence
                    and previous.class_confidence
                    > self.config.tracking.high_class_confidence
                ):
                    continue
                appearance = torch.dot(
                    item.representative_embedding,
                    previous.representative_embedding,
                )
                if float(appearance) < self.config.tracking.clip_threshold:
                    continue
                score = (
                    self.config.tracking.position_weight
                    * (1.0 - distance / self.config.tracking.centroid_gate_m)
                    + self.config.tracking.appearance_weight * appearance
                )
                candidates.append(
                    (float(score), current_index, previous_index)
                )

        used_current: set[int] = set()
        used_previous: set[int] = set()
        for _, current_index, previous_index in sorted(
            candidates,
            reverse=True,
        ):
            if current_index in used_current or previous_index in used_previous:
                continue
            current[current_index].track_id = (
                self.previous_objects[previous_index].track_id
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
        speeds: list[float] = []
        ious: list[float] = []

        for lag in self.config.tracking.history_lags:
            old = track.history.get(frame_index - lag)
            if old is None:
                continue
            dt_s = (stamp_ns - old.stamp_ns) * 1.0e-9
            if dt_s <= 0.0:
                continue
            displacement = torch.linalg.norm(
                item.centroid_world - old.centroid_world
            )
            speeds.append(float(displacement) / dt_s)
            ious.append(float(voxel_iou(item.voxel_keys, old.voxel_keys)))

        if not speeds:
            return MotionState.STATIC

        median_speed = statistics.median(speeds)
        median_iou = statistics.median(ious)
        return (
            MotionState.MOVING
            if (
                median_speed
                > self.config.tracking.centroid_speed_threshold_mps
                or median_iou
                < self.config.tracking.voxel_iou_threshold
            )
            else MotionState.STATIC
        )

    def update(self, frame: VoxelizedFrame) -> TrackedFrame:
        self._assign_tracks(frame.objects)

        for item in frame.objects:
            item.motion_state = self._classify_motion(
                item,
                frame.frame_index,
                frame.stamp_ns,
            )
            track = self.tracks[item.track_id]
            track.last_object = item
            track.history[frame.frame_index] = TrackObservation(
                frame_index=frame.frame_index,
                stamp_ns=frame.stamp_ns,
                centroid_world=item.centroid_world,
                voxel_keys=item.voxel_keys,
                points=item.points,
            )
            oldest = frame.frame_index - max(
                self.config.tracking.history_lags
            ) - 1
            for key in tuple(track.history):
                if key < oldest:
                    del track.history[key]

        moving_members: dict[str, list[torch.Tensor]] = defaultdict(list)
        for item in frame.objects:
            if item.motion_state != MotionState.MOVING:
                continue
            for member in item.members:
                moving_members[member.camera_name].append(
                    member.mask_original
                )

        moving_masks: dict[str, torch.Tensor] = {}
        for camera_name, result in frame.camera_results.items():
            shape = result.camera.depth.shape
            masks = moving_members.get(camera_name, [])
            if masks:
                moving_masks[camera_name] = torch.stack(masks).any(dim=0)
            else:
                moving_masks[camera_name] = torch.zeros(
                    shape,
                    device=result.camera.depth.device,
                    dtype=torch.bool,
                )

        self.previous_objects = frame.objects
        return TrackedFrame(
            frame_index=frame.frame_index,
            stamp_ns=frame.stamp_ns,
            objects=frame.objects,
            background_points=frame.background_points,
            background_keys=frame.background_keys,
            moving_masks=moving_masks,
        )
