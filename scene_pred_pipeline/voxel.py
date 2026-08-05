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
_NEIGHBOR_OFFSETS: dict[
    tuple[int, str],
    torch.Tensor,
] = {}


def pack_voxel_coords(
    coords: torch.Tensor,
) -> torch.Tensor:
    coords = coords.to(torch.int64)
    shifted = coords + _KEY_BIAS
    if bool(
        (
            (shifted < 0)
            | (shifted > _KEY_MASK)
        ).any()
    ):
        raise ValueError(
            "Voxel coordinates exceed the supported "
            "21-bit range."
        )
    return (
        shifted[:, 0] << (2 * _KEY_BITS)
    ) | (
        shifted[:, 1] << _KEY_BITS
    ) | shifted[:, 2]


def unpack_voxel_keys(
    keys: torch.Tensor,
) -> torch.Tensor:
    keys = keys.to(torch.int64)
    x = (
        (keys >> (2 * _KEY_BITS))
        & _KEY_MASK
    ) - _KEY_BIAS
    y = (
        (keys >> _KEY_BITS)
        & _KEY_MASK
    ) - _KEY_BIAS
    z = (
        keys & _KEY_MASK
    ) - _KEY_BIAS
    return torch.stack((x, y, z), dim=1)


def voxel_coords(
    points: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor,
) -> torch.Tensor:
    return torch.floor(
        (points - origin) / voxel_size
    ).to(torch.int64)


def voxel_keys(
    points: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor,
) -> torch.Tensor:
    return pack_voxel_coords(
        voxel_coords(
            points,
            voxel_size,
            origin,
        )
    )


def voxel_downsample(
    points: torch.Tensor,
    voxel_size: float,
    origin: torch.Tensor,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    if points.shape[0] == 0:
        return (
            points,
            torch.empty(
                (0,),
                device=points.device,
                dtype=torch.int64,
            ),
            torch.empty(
                (0, 3),
                device=points.device,
                dtype=torch.int64,
            ),
        )

    keys = voxel_keys(
        points,
        voxel_size,
        origin,
    )
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
    sampled = sums / counts.clamp_min_(1.0)
    coords = unpack_voxel_keys(unique_keys)
    return sampled, unique_keys, coords


def voxel_iou(
    first: torch.Tensor,
    second: torch.Tensor,
) -> torch.Tensor:
    if (
        first.numel() == 0
        and second.numel() == 0
    ):
        return torch.ones(
            (),
            device=first.device,
        )
    if (
        first.numel() == 0
        or second.numel() == 0
    ):
        return torch.zeros(
            (),
            device=first.device,
        )

    combined = torch.cat(
        (first, second)
    )
    union_count = torch.unique(
        combined
    ).numel()
    intersection_count = (
        first.numel()
        + second.numel()
        - union_count
    )
    return torch.tensor(
        intersection_count
        / max(1, union_count),
        device=first.device,
        dtype=torch.float32,
    )


def _neighbor_offsets(
    radius: int,
    device: torch.device,
) -> torch.Tensor:
    key = (int(radius), str(device))
    cached = _NEIGHBOR_OFFSETS.get(key)
    if cached is not None:
        return cached

    values = torch.arange(
        -radius,
        radius + 1,
        device=device,
        dtype=torch.int64,
    )
    offsets = torch.cartesian_prod(
        values,
        values,
        values,
    )
    _NEIGHBOR_OFFSETS[key] = offsets
    return offsets


def voxel_neighbor_coverage(
    query_keys: torch.Tensor,
    reference_coords: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Fraction of query voxels inside a dilated reference voxel set."""

    if (
        query_keys.numel() == 0
        or reference_coords.numel() == 0
    ):
        return torch.zeros(
            (),
            device=query_keys.device,
        )
    if radius <= 0:
        reference_keys = pack_voxel_coords(
            reference_coords
        )
    else:
        offsets = _neighbor_offsets(
            radius,
            reference_coords.device,
        )
        expanded = (
            reference_coords[:, None, :]
            + offsets[None, :, :]
        )
        reference_keys = torch.unique(
            pack_voxel_coords(
                expanded.reshape(-1, 3)
            )
        )
    return torch.isin(
        query_keys,
        reference_keys,
    ).float().mean()


class GlobalVoxelizer:
    """Fixed world-frame voxel downsampling."""

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config
        self.origin: torch.Tensor | None = None

    def process(
        self,
        frame_index: int,
        fused: FusedFrame,
    ) -> VoxelizedFrame:
        device = fused.background_pcd_world.device
        if (
            self.origin is None
            or self.origin.device != device
        ):
            self.origin = torch.tensor(
                self.config.voxel.origin_world,
                device=device,
                dtype=torch.float32,
            )

        objects: list[VoxelizedObject] = []
        for item in fused.objects:
            points, keys, coords = voxel_downsample(
                item.pcd_world,
                self.config.voxel.size_m,
                self.origin,
            )
            if points.shape[0] == 0:
                continue
            objects.append(
                VoxelizedObject(
                    frame_object_id=item.frame_object_id,
                    class_id=item.class_id,
                    class_name=item.class_name,
                    class_confidence=item.class_confidence,
                    points=points,
                    voxel_keys=keys,
                    voxel_coords=coords,
                    centroid_world=points.median(
                        dim=0
                    ).values,
                    aabb_min_world=points.amin(
                        dim=0
                    ),
                    aabb_max_world=points.amax(
                        dim=0
                    ),
                    members=item.members,
                )
            )

        (
            background_points,
            background_keys,
            _,
        ) = voxel_downsample(
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
