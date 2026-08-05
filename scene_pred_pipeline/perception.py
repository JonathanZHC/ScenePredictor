from __future__ import annotations

import math
from contextlib import nullcontext
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .profiler import CycleProfiler

import open_clip
import torch
import torch.nn.functional as F
from ultralytics import YOLO

from .config import PipelineConfig
from .data_types import (
    CameraFrameGpu,
    MultiCameraFrame,
    PerViewResult,
    ViewInstance,
)


def _profile_stage(
    profiler: "CycleProfiler | None",
    name: str,
):
    if profiler is None:
        return nullcontext()
    return profiler.stage(name)


def _stamp_to_gpu(frame, device: torch.device) -> CameraFrameGpu:
    rgb = torch.from_numpy(frame.rgb).to(
        device=device,
        dtype=torch.uint8,
        non_blocking=True,
    )
    depth = torch.from_numpy(frame.depth).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    K = torch.from_numpy(frame.K).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    T_world_camera = torch.from_numpy(frame.T_world_camera).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return CameraFrameGpu(
        camera_name=frame.camera_name,
        stamp_ns=frame.stamp_ns,
        rgb=rgb,
        depth=depth,
        K=K,
        T_world_camera=T_world_camera,
        T_camera_world=torch.linalg.inv(T_world_camera),
        optical_frame_id=frame.optical_frame_id,
    )


def _binary_dilate(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    kernel = 2 * pixels + 1
    return (
        F.max_pool2d(
            mask.float()[None, None],
            kernel_size=kernel,
            stride=1,
            padding=pixels,
        )[0, 0]
        > 0.5
    )


def _binary_erode(mask: torch.Tensor, pixels: int) -> torch.Tensor:
    if pixels <= 0:
        return mask
    return ~_binary_dilate(~mask, pixels)


def _masked_points_world(
    camera: CameraFrameGpu,
    mask: torch.Tensor,
    min_depth: float,
    max_depth: float,
) -> torch.Tensor:
    depth = camera.depth
    valid = (
        mask
        & torch.isfinite(depth)
        & (depth >= min_depth)
        & (depth <= max_depth)
    )
    v, u = torch.where(valid)
    if u.numel() == 0:
        return torch.empty((0, 3), device=depth.device, dtype=torch.float32)

    z = depth[v, u]
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy
    camera_points = torch.stack((x, y, z), dim=1)

    rotation = camera.T_world_camera[:3, :3]
    translation = camera.T_world_camera[:3, 3]
    return camera_points @ rotation.T + translation


def _uniform_points(points: torch.Tensor, count: int) -> torch.Tensor:
    if points.shape[0] <= count:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        count,
        device=points.device,
    ).long()
    return points[indices]


class PerViewPerception:
    """Step 1: YOLO, masks, RGB-D back-projection and CLIP embeddings."""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.yolo = YOLO(config.models.yolo_weights)
        self.yolo.to(str(self.device))

        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            config.models.clip_model,
            pretrained=config.models.clip_pretrained,
            device=self.device,
        )
        self.clip_model.eval()

        visual = self.clip_model.visual
        image_size = getattr(visual, "image_size", 224)
        if isinstance(image_size, tuple):
            self.clip_size = tuple(int(value) for value in image_size)
        else:
            self.clip_size = (int(image_size), int(image_size))

        mean = getattr(visual, "image_mean", (0.48145466, 0.4578275, 0.40821073))
        std = getattr(visual, "image_std", (0.26862954, 0.26130258, 0.27577711))
        self.clip_mean = torch.tensor(mean, device=self.device).view(1, 3, 1, 1)
        self.clip_std = torch.tensor(std, device=self.device).view(1, 3, 1, 1)

    def _prepare_clip_crop(
        self,
        rgb: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        ys, xs = torch.where(mask)
        if xs.numel() == 0:
            return torch.zeros(
                (3, *self.clip_size),
                device=self.device,
                dtype=torch.float32,
            )

        height, width = mask.shape
        x0, x1 = int(xs.min()), int(xs.max()) + 1
        y0, y1 = int(ys.min()), int(ys.max()) + 1
        pad_x = int(math.ceil((x1 - x0) * self.config.models.clip_crop_padding))
        pad_y = int(math.ceil((y1 - y0) * self.config.models.clip_crop_padding))
        x0, x1 = max(0, x0 - pad_x), min(width, x1 + pad_x)
        y0, y1 = max(0, y0 - pad_y), min(height, y1 + pad_y)

        crop_rgb = rgb[y0:y1, x0:x1].permute(2, 0, 1).float() / 255.0
        crop_mask = mask[y0:y1, x0:x1].float()[None]
        neutral = torch.full_like(crop_rgb, 0.5)
        crop = crop_rgb * crop_mask + neutral * (1.0 - crop_mask)
        return F.interpolate(
            crop[None],
            size=self.clip_size,
            mode="bilinear",
            align_corners=False,
        )[0]

    def process(
        self,
        frame: MultiCameraFrame,
        profiler: "CycleProfiler | None" = None,
    ) -> dict[str, PerViewResult]:

        with _profile_stage(profiler, "step1_h2d"):
            camera_gpu = {
                name: _stamp_to_gpu(value, self.device)
                for name, value in frame.cameras.items()
            }

            ordered_names = list(camera_gpu)

            rgb_batch = torch.stack(
                [
                    camera_gpu[name].rgb.permute(2, 0, 1)
                    for name in ordered_names
                ]
            ).float().div_(255.0)

        with _profile_stage(profiler, "step1_yolo"):
            yolo_results = self.yolo.predict(
                source=rgb_batch,
                imgsz=self.config.models.yolo_image_size,
                conf=self.config.models.yolo_confidence,
                iou=self.config.models.yolo_iou,
                device=str(self.device),
                verbose=False,
                retina_masks=True,
            )

        pending_clip: list[tuple[ViewInstance, torch.Tensor, torch.Tensor]] = []
        output: dict[str, PerViewResult] = {}

        with _profile_stage(profiler, "step1_mask_geometry"):

            for camera_name, result in zip(
                ordered_names,
                yolo_results,
                strict=True,
            ):
                camera = camera_gpu[camera_name]
                height, width = camera.depth.shape

                instances: list[ViewInstance] = []

                exclusion = torch.zeros(
                    (height, width),
                    device=self.device,
                    dtype=torch.bool,
                )

                if result.masks is not None:
                    masks = result.masks.data.to(
                        self.device,
                        non_blocking=True,
                    )

                    if masks.shape[-2:] != (height, width):
                        masks = F.interpolate(
                            masks[:, None].float(),
                            size=(height, width),
                            mode="nearest",
                        )[:, 0]

                    masks = masks > 0.5

                    classes_cpu = (
                        result.boxes.cls
                        .to(torch.int64)
                        .cpu()
                        .tolist()
                    )
                    confidences_cpu = (
                        result.boxes.conf
                        .float()
                        .cpu()
                        .tolist()
                    )

                    for local_id in range(masks.shape[0]):
                        original = masks[local_id]

                        eroded = _binary_erode(
                            original,
                            self.config.masks.object_erosion_pixels,
                        )

                        exclusion |= _binary_dilate(
                            original,
                            self.config.masks.background_dilation_pixels,
                        )

                        points = _masked_points_world(
                            camera,
                            eroded,
                            self.config.depth.min_m,
                            self.config.depth.max_m,
                        )

                        if points.shape[0] == 0:
                            continue

                        centroid = points.mean(dim=0)

                        sparse = _uniform_points(
                            points,
                            self.config.multiview.reprojection_points,
                        )

                        placeholder = torch.empty(
                            (0,),
                            device=self.device,
                            dtype=torch.float32,
                        )

                        instance = ViewInstance(
                            camera_name=camera_name,
                            local_instance_id=local_id,
                            class_id=classes_cpu[local_id],
                            class_confidence=confidences_cpu[local_id],
                            mask_original=original,
                            mask_eroded=eroded,
                            clip_embedding=placeholder,
                            pcd_world=points,
                            centroid_world=centroid,
                            reprojection_points_world=sparse,
                        )

                        pending_clip.append(
                            (
                                instance,
                                camera.rgb,
                                original,
                            )
                        )
                        instances.append(instance)

                background_mask = ~exclusion

                background_points = _masked_points_world(
                    camera,
                    background_mask,
                    self.config.depth.min_m,
                    self.config.depth.max_m,
                )

                output[camera_name] = PerViewResult(
                    camera=camera,
                    instances=instances,
                    background_pcd_world=background_points,
                )

        crops: torch.Tensor | None = None

        if pending_clip:
            with _profile_stage(profiler, "step1_clip_preprocess"):
                crops = torch.stack(
                    [
                        self._prepare_clip_crop(
                            rgb,
                            mask,
                        )
                        for _, rgb, mask in pending_clip
                    ]
                )                

        if pending_clip and crops is not None:
            with _profile_stage(profiler, "step1_clip_encode"):
                embeddings: list[torch.Tensor] = []

                batch_size = (
                    self.config.models.clip_batch_size
                )

                with torch.inference_mode(), torch.autocast(
                    device_type="cuda",
                    dtype=torch.float16,
                    enabled=self.device.type == "cuda",
                ):
                    for begin in range(
                        0,
                        crops.shape[0],
                        batch_size,
                    ):
                        batch = crops[
                            begin : begin + batch_size
                        ]

                        batch = (
                            batch - self.clip_mean
                        ) / self.clip_std

                        encoded = self.clip_model.encode_image(
                            batch
                        )

                        encoded = F.normalize(
                            encoded.float(),
                            dim=1,
                        )

                        embeddings.extend(
                            encoded.unbind(dim=0)
                        )

                for (
                    instance,
                    _,
                    _,
                ), embedding in zip(
                    pending_clip,
                    embeddings,
                    strict=True,
                ):
                    instance.clip_embedding = embedding

        return output
