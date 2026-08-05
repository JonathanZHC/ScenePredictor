from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch

from .config import PipelineConfig
from .data_types import (
    FusedFrame,
    FusedObject,
    PerViewResult,
    ViewInstance,
)


class _UnionFind:
    """Graph components with one local instance per camera."""

    def __init__(
        self,
        count: int,
        camera_names: list[str],
    ) -> None:
        self.parent = list(range(count))
        self.cameras = [
            {camera_names[index]}
            for index in range(count)
        ]

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[
                self.parent[index]
            ]
            index = self.parent[index]
        return index

    def union(
        self,
        first: int,
        second: int,
    ) -> bool:
        root_a = self.find(first)
        root_b = self.find(second)
        if root_a == root_b:
            return False
        if self.cameras[root_a] & self.cameras[root_b]:
            return False

        if len(self.cameras[root_a]) < len(
            self.cameras[root_b]
        ):
            root_a, root_b = root_b, root_a

        self.parent[root_b] = root_a
        self.cameras[root_a] |= self.cameras[root_b]
        return True


def _aabb_gap_numpy(
    min_a: np.ndarray,
    max_a: np.ndarray,
    min_b: np.ndarray,
    max_b: np.ndarray,
) -> float:
    separation = np.maximum(
        np.maximum(min_a - max_b, min_b - max_a),
        0.0,
    )
    return float(np.linalg.norm(separation))


def _project_counts(
    source: ViewInstance,
    target: ViewInstance,
    target_view: PerViewResult,
    config: PipelineConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Count depth-consistent source points landing in the target mask.

    Points hidden behind the target camera's measured surface are ignored,
    rather than treated as negative evidence.
    """

    points_world = source.reprojection_points_world
    device = target_view.camera.depth.device
    if points_world.shape[0] == 0:
        zero = torch.zeros(
            (),
            device=device,
            dtype=torch.int64,
        )
        return zero, zero

    rotation = target_view.camera.T_camera_world[
        :3, :3
    ]
    translation = target_view.camera.T_camera_world[
        :3, 3
    ]
    points_camera = (
        points_world @ rotation.T + translation
    )
    x, y, z = points_camera.unbind(dim=1)

    K = target_view.camera.K
    safe_z = z.clamp_min(1.0e-6)
    u = torch.round(
        K[0, 0] * x / safe_z + K[0, 2]
    ).long()
    v = torch.round(
        K[1, 1] * y / safe_z + K[1, 2]
    ).long()

    height, width = target_view.camera.depth.shape
    in_image = (
        (z > 0.0)
        & (u >= 0)
        & (u < width)
        & (v >= 0)
        & (v < height)
    )

    u_safe = u.clamp(0, width - 1)
    v_safe = v.clamp(0, height - 1)
    observed_depth = target_view.camera.depth[
        v_safe,
        u_safe,
    ]
    mask_hit = target.mask_original[
        v_safe,
        u_safe,
    ]

    valid_depth = (
        torch.isfinite(observed_depth)
        & (observed_depth > 0.0)
    )
    valid = in_image & valid_depth
    occluded = valid & (
        z
        > observed_depth
        + config.multiview.occlusion_tolerance_m
    )
    visible = valid & ~occluded
    depth_consistent = (
        torch.abs(z - observed_depth)
        <= config.multiview.depth_tolerance_m
    )
    matched = visible & mask_hit & depth_consistent
    return matched.sum(), visible.sum()


class MultiViewFusion:
    """Class-gated graph association from RGB-D reprojection geometry."""

    def __init__(
        self,
        config: PipelineConfig,
    ) -> None:
        self.config = config

    def process(
        self,
        stamp_ns: int,
        views: dict[str, PerViewResult],
    ) -> FusedFrame:
        instances = [
            instance
            for view in views.values()
            for instance in view.instances
        ]
        camera_names = [
            instance.camera_name
            for instance in instances
        ]
        union_find = _UnionFind(
            len(instances),
            camera_names,
        )

        # One compact D2H transfer for all robust centers and AABBs. Dense
        # masks, depth maps, point clouds and reprojection remain on the GPU.
        if instances:
            geometry = torch.stack(
                [
                    torch.cat(
                        (
                            item.centroid_world,
                            item.aabb_min_world,
                            item.aabb_max_world,
                        )
                    )
                    for item in instances
                ]
            ).detach().float().cpu().numpy()
        else:
            geometry = np.empty((0, 9), dtype=np.float32)

        pending: list[
            tuple[
                int,
                int,
                float,
                float,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ]
        ] = []
        for first in range(len(instances)):
            a = instances[first]
            for second in range(
                first + 1,
                len(instances),
            ):
                b = instances[second]
                if a.camera_name == b.camera_name:
                    continue
                if a.class_id != b.class_id:
                    continue

                centroid_distance = float(
                    np.linalg.norm(
                        geometry[first, :3]
                        - geometry[second, :3]
                    )
                )
                box_gap = _aabb_gap_numpy(
                    geometry[first, 3:6],
                    geometry[first, 6:9],
                    geometry[second, 3:6],
                    geometry[second, 6:9],
                )
                if not (
                    centroid_distance
                    <= self.config.multiview.centroid_gate_m
                    or box_gap
                    <= self.config.multiview.aabb_gap_gate_m
                ):
                    continue

                match_ab, visible_ab = _project_counts(
                    a,
                    b,
                    views[b.camera_name],
                    self.config,
                )
                match_ba, visible_ba = _project_counts(
                    b,
                    a,
                    views[a.camera_name],
                    self.config,
                )
                pending.append(
                    (
                        first,
                        second,
                        centroid_distance,
                        box_gap,
                        match_ab,
                        visible_ab,
                        match_ba,
                        visible_ba,
                    )
                )

        candidates: list[
            tuple[float, int, int]
        ] = []
        if pending:
            # One scalar-table D2H transfer for every candidate edge.
            counts = torch.stack(
                [
                    torch.stack(
                        (
                            item[4],
                            item[5],
                            item[6],
                            item[7],
                        )
                    )
                    for item in pending
                ]
            ).detach().cpu().numpy()

            for item, values in zip(
                pending,
                counts,
                strict=True,
            ):
                (
                    first,
                    second,
                    centroid_distance,
                    box_gap,
                    *_unused,
                ) = item
                match_ab, visible_ab, match_ba, visible_ba = (
                    int(value) for value in values
                )
                total_visible = visible_ab + visible_ba
                if (
                    total_visible
                    < self.config.multiview.minimum_visible_points
                ):
                    continue

                reprojection = (
                    match_ab + match_ba
                ) / total_visible
                if (
                    reprojection
                    < self.config.multiview.reprojection_threshold
                ):
                    continue

                normalized_gap = min(
                    centroid_distance
                    / max(
                        self.config.multiview.centroid_gate_m,
                        1.0e-6,
                    ),
                    box_gap
                    / max(
                        self.config.multiview.aabb_gap_gate_m,
                        1.0e-6,
                    ),
                    1.0,
                )
                spatial_score = 1.0 - normalized_gap
                score = (
                    self.config.multiview.reprojection_weight
                    * reprojection
                    + self.config.multiview.spatial_weight
                    * spatial_score
                )
                candidates.append(
                    (score, first, second)
                )

        # Strongest compatible graph edges are accepted first.
        for _, first, second in sorted(
            candidates,
            reverse=True,
        ):
            union_find.union(first, second)

        groups: dict[
            int,
            list[ViewInstance],
        ] = defaultdict(list)
        for index, instance in enumerate(instances):
            groups[
                union_find.find(index)
            ].append(instance)

        fused_objects: list[FusedObject] = []
        for object_id, members in enumerate(
            groups.values()
        ):
            points = torch.cat(
                [
                    member.pcd_world
                    for member in members
                ],
                dim=0,
            )
            best_member = max(
                members,
                key=lambda value: (
                    value.class_confidence
                ),
            )
            fused_objects.append(
                FusedObject(
                    frame_object_id=object_id,
                    class_id=best_member.class_id,
                    class_name=best_member.class_name,
                    class_confidence=float(
                        sum(
                            member.class_confidence
                            for member in members
                        )
                        / len(members)
                    ),
                    pcd_world=points,
                    centroid_world=points.median(
                        dim=0
                    ).values,
                    aabb_min_world=points.amin(
                        dim=0
                    ),
                    aabb_max_world=points.amax(
                        dim=0
                    ),
                    source_camera_names=tuple(
                        member.camera_name
                        for member in members
                    ),
                    members=members,
                )
            )

        background_parts = [
            view.background_pcd_world
            for view in views.values()
            if view.background_pcd_world.numel() > 0
        ]
        if background_parts:
            background = torch.cat(
                background_parts,
                dim=0,
            )
        else:
            device = next(
                iter(views.values())
            ).camera.depth.device
            background = torch.empty(
                (0, 3),
                device=device,
            )

        return FusedFrame(
            stamp_ns=stamp_ns,
            objects=fused_objects,
            background_pcd_world=background,
            camera_results=views,
        )
