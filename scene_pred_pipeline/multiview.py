from __future__ import annotations

from collections import defaultdict

import torch

from .config import PipelineConfig
from .data_types import FusedFrame, FusedObject, PerViewResult, ViewInstance


class _UnionFind:
    def __init__(self, count: int, camera_names: list[str]) -> None:
        self.parent = list(range(count))
        self.cameras = [{camera_names[index]} for index in range(count)]

    def find(self, index: int) -> int:
        while self.parent[index] != index:
            self.parent[index] = self.parent[self.parent[index]]
            index = self.parent[index]
        return index

    def union(self, first: int, second: int) -> bool:
        root_a, root_b = self.find(first), self.find(second)
        if root_a == root_b:
            return False
        if self.cameras[root_a] & self.cameras[root_b]:
            return False
        self.parent[root_b] = root_a
        self.cameras[root_a] |= self.cameras[root_b]
        return True


def _project_score(
    source: ViewInstance,
    target: ViewInstance,
    target_view: PerViewResult,
    config: PipelineConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    points_world = source.reprojection_points_world
    if points_world.shape[0] == 0:
        zero = torch.zeros((), device=target_view.camera.depth.device)
        return zero, zero

    rotation = target_view.camera.T_camera_world[:3, :3]
    translation = target_view.camera.T_camera_world[:3, 3]
    points_camera = points_world @ rotation.T + translation

    x, y, z = points_camera.unbind(dim=1)
    K = target_view.camera.K
    u = torch.round(K[0, 0] * x / z + K[0, 2]).long()
    v = torch.round(K[1, 1] * y / z + K[1, 2]).long()
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
    observed_depth = target_view.camera.depth[v_safe, u_safe]
    mask_hit = target.mask_original[v_safe, u_safe]
    valid_depth = torch.isfinite(observed_depth) & (observed_depth > 0.0)
    valid = in_image & valid_depth

    occluded = valid & (
        z > observed_depth + config.multiview.occlusion_tolerance_m
    )
    visible = valid & ~occluded
    depth_consistent = (
        torch.abs(z - observed_depth)
        < config.multiview.depth_tolerance_m
    )
    matched = visible & mask_hit & depth_consistent
    return matched.sum(), visible.sum()


def _medoid(embeddings: list[torch.Tensor]) -> torch.Tensor:
    if len(embeddings) == 1:
        return embeddings[0]
    matrix = torch.stack(embeddings)
    similarity = matrix @ matrix.T
    return matrix[similarity.sum(dim=1).argmax()]


class MultiViewFusion:
    """Step 2: centroid gate, sparse reprojection and greedy fusion."""

    def __init__(self, config: PipelineConfig) -> None:
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
        camera_names = [instance.camera_name for instance in instances]
        union_find = _UnionFind(len(instances), camera_names)
        candidates: list[tuple[float, int, int]] = []

        for first in range(len(instances)):
            a = instances[first]
            for second in range(first + 1, len(instances)):
                b = instances[second]
                if a.camera_name == b.camera_name:
                    continue

                centroid_distance = torch.linalg.norm(
                    a.centroid_world - b.centroid_world
                )
                if float(centroid_distance) > self.config.multiview.centroid_gate_m:
                    continue

                if (
                    a.class_id != b.class_id
                    and a.class_confidence
                    > self.config.multiview.high_class_confidence
                    and b.class_confidence
                    > self.config.multiview.high_class_confidence
                ):
                    continue

                match_ab, visible_ab = _project_score(
                    a,
                    b,
                    views[b.camera_name],
                    self.config,
                )
                match_ba, visible_ba = _project_score(
                    b,
                    a,
                    views[a.camera_name],
                    self.config,
                )
                total_visible = visible_ab + visible_ba
                if int(total_visible) < self.config.multiview.minimum_visible_points:
                    continue

                reprojection = (match_ab + match_ba).float() / total_visible.float()
                clip_similarity = torch.dot(
                    a.clip_embedding,
                    b.clip_embedding,
                )

                if float(reprojection) < self.config.multiview.reprojection_threshold:
                    continue
                if float(clip_similarity) < self.config.multiview.clip_threshold:
                    continue

                normalized_distance = (
                    centroid_distance / self.config.multiview.centroid_gate_m
                )
                score = (
                    self.config.multiview.reprojection_weight * reprojection
                    + self.config.multiview.clip_weight * clip_similarity
                    - self.config.multiview.centroid_weight * normalized_distance
                )
                candidates.append((float(score), first, second))

        for _, first, second in sorted(candidates, reverse=True):
            union_find.union(first, second)

        groups: dict[int, list[ViewInstance]] = defaultdict(list)
        for index, instance in enumerate(instances):
            groups[union_find.find(index)].append(instance)

        fused_objects: list[FusedObject] = []
        for object_id, members in enumerate(groups.values()):
            class_scores: dict[int, float] = defaultdict(float)
            for member in members:
                class_scores[member.class_id] += member.class_confidence
            class_id = max(class_scores, key=class_scores.get)
            class_score = class_scores[class_id] / max(
                1,
                sum(member.class_id == class_id for member in members),
            )
            points = torch.cat([member.pcd_world for member in members], dim=0)
            embeddings = [member.clip_embedding for member in members]
            fused_objects.append(
                FusedObject(
                    frame_object_id=object_id,
                    class_id=class_id,
                    class_confidence=float(class_score),
                    representative_embedding=_medoid(embeddings),
                    view_embeddings=embeddings,
                    pcd_world=points,
                    centroid_world=points.mean(dim=0),
                    source_camera_names=tuple(
                        member.camera_name for member in members
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
            background = torch.cat(background_parts, dim=0)
        else:
            device = next(iter(views.values())).camera.depth.device
            background = torch.empty((0, 3), device=device)

        return FusedFrame(
            stamp_ns=stamp_ns,
            objects=fused_objects,
            background_pcd_world=background,
            camera_results=views,
        )
