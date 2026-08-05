from __future__ import annotations

import torch

from .config import PipelineConfig
from .data_types import (
    FusedFrame,
    VoxelizedFrame,
    VoxelizedObject,
)


_KEY_BITS = 21
_KEY_BIAS = 1 << (_KEY_BITS - 1)
_KEY_MASK = (1 << _KEY_BITS) - 1


def voxel_keys(
    points: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor,
) -> torch.Tensor:
    coords = torch.floor((points - origin) / voxel_size).to(torch.int64)
    shifted = coords + _KEY_BIAS
    if bool(((shifted < 0) | (shifted > _KEY_MASK)).any()):
        raise ValueError("Voxel coordinates exceed the supported 21-bit range.")
    return (
        (shifted[:, 0] << (2 * _KEY_BITS))
        | (shifted[:, 1] << _KEY_BITS)
        | shifted[:, 2]
    )


def voxel_downsample(
    points: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if points.shape[0] == 0:
        return points, torch.empty(
            (0,),
            device=points.device,
            dtype=torch.int64,
        )

    keys = voxel_keys(points, voxel_size, origin)
    unique_keys, inverse = torch.unique(
        keys,
        sorted=False,
        return_inverse=True,
    )
    sums = torch.zeros(
        (unique_keys.shape[0], 3),
        device=points.device,
        dtype=points.dtype,
    )
    counts = torch.zeros(
        (unique_keys.shape[0], 1),
        device=points.device,
        dtype=points.dtype,
    )
    sums.index_add_(0, inverse, points)
    counts.index_add_(
        0,
        inverse,
        torch.ones(
            (points.shape[0], 1),
            device=points.device,
            dtype=points.dtype,
        ),
    )
    return sums / counts.clamp_min_(1.0), unique_keys


def voxel_iou(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    if first.numel() == 0 and second.numel() == 0:
        return torch.ones((), device=first.device)
    combined = torch.cat((first, second))
    union_count = torch.unique(combined).numel()
    intersection_count = (
        first.numel() + second.numel() - union_count
    )
    return torch.tensor(
        intersection_count / max(1, union_count),
        device=first.device,
        dtype=torch.float32,
    )


class GlobalVoxelizer:
    """Step 3: fixed world-frame voxel downsampling."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.origin: torch.Tensor | None = None

    def process(
        self,
        frame_index: int,
        fused: FusedFrame,
    ) -> VoxelizedFrame:
        device = fused.background_pcd_world.device
        if self.origin is None or self.origin.device != device:
            self.origin = torch.tensor(
                self.config.voxel.origin_world,
                device=device,
                dtype=torch.float32,
            )

        objects: list[VoxelizedObject] = []
        for item in fused.objects:
            points, keys = voxel_downsample(
                item.pcd_world,
                self.config.voxel.size_m,
                self.origin,
            )
            objects.append(
                VoxelizedObject(
                    frame_object_id=item.frame_object_id,
                    class_id=item.class_id,
                    class_confidence=item.class_confidence,
                    representative_embedding=item.representative_embedding,
                    points=points,
                    voxel_keys=keys,
                    centroid_world=points.mean(dim=0),
                    members=item.members,
                )
            )

        background_points, background_keys = voxel_downsample(
            fused.background_pcd_world,
            self.config.voxel.size_m,
            self.origin,
        )
        return VoxelizedFrame(
            frame_index=frame_index,
            stamp_ns=fused.stamp_ns,
            objects=objects,
            background_points=background_points,
            background_keys=background_keys,
            camera_results=fused.camera_results,
        )
