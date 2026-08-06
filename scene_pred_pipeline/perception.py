from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from ultralytics import YOLO, YOLOE

from .config import PipelineConfig
from .data_types import (
    CameraFrameGpu,
    MultiCameraFrame,
    PerViewResult,
    ViewInstance,
)
from .labels import load_object_labels

if TYPE_CHECKING:
    from .profiler import CycleProfiler


def _rgb8_to_ultralytics_bgr(
    image: np.ndarray,
) -> np.ndarray:
    """Convert ROS rgb8 image to Ultralytics NumPy BGR input."""

    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"Expected NumPy image, got {type(image).__name__}."
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected HxWx3 RGB image, got {image.shape}."
        )

    if image.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 image, got {image.dtype}."
        )

    # RGB -> BGR and remove the negative stride.
    return np.ascontiguousarray(
        image[..., ::-1]
    )


def _frame_to_gpu(
    frame,
    device: torch.device,
) -> CameraFrameGpu:
    """Transfer only geometry inputs.

    RGB is passed once to Ultralytics as a NumPy batch. It is not copied to a
    second persistent GPU tensor.
    """

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
    T_world_camera = torch.from_numpy(
        frame.T_world_camera
    ).to(
        device=device,
        dtype=torch.float32,
        non_blocking=True,
    )
    return CameraFrameGpu(
        camera_name=frame.camera_name,
        stamp_ns=frame.stamp_ns,
        depth=depth,
        K=K,
        T_world_camera=T_world_camera,
        T_camera_world=torch.linalg.inv(T_world_camera),
        optical_frame_id=frame.optical_frame_id,
    )


def _binary_dilate(
    masks: torch.Tensor,
    pixels: int,
) -> torch.Tensor:
    if pixels <= 0 or masks.numel() == 0:
        return masks
    kernel = 2 * pixels + 1
    return (
        F.max_pool2d(
            masks.float()[:, None],
            kernel_size=kernel,
            stride=1,
            padding=pixels,
        )[:, 0]
        > 0.5
    )


def _binary_erode(
    masks: torch.Tensor,
    pixels: int,
) -> torch.Tensor:
    if pixels <= 0 or masks.numel() == 0:
        return masks
    return ~_binary_dilate(~masks, pixels)


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
        return torch.empty(
            (0, 3),
            device=depth.device,
            dtype=torch.float32,
        )

    z = depth[v, u]
    fx, fy = camera.K[0, 0], camera.K[1, 1]
    cx, cy = camera.K[0, 2], camera.K[1, 2]
    x = (u.float() - cx) * z / fx
    y = (v.float() - cy) * z / fy
    points_camera = torch.stack((x, y, z), dim=1)

    rotation = camera.T_world_camera[:3, :3]
    translation = camera.T_world_camera[:3, 3]
    return points_camera @ rotation.T + translation


def _uniform_points(
    points: torch.Tensor,
    count: int,
) -> torch.Tensor:
    if points.shape[0] <= count:
        return points
    indices = torch.linspace(
        0,
        points.shape[0] - 1,
        count,
        device=points.device,
    ).long()
    return points[indices]


def _model_names(result) -> tuple[str, ...]:
    names = result.names
    if isinstance(names, dict):
        return tuple(
            str(names[index])
            for index in sorted(names)
        )
    return tuple(str(value) for value in names)


class PerViewPerception:
    """YOLOE instance segmentation plus RGB-D geometry extraction.

    TensorRT inference uses a fixed vocabulary embedded at export time.
    labels are always read from models.label_file, never from YAML lists.
    """

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = torch.device(config.runtime.device)
        self.backend = config.models.backend.strip().lower()
        self.labels = load_object_labels(
            config.models.label_file
        )
        self.engine_batch_size = max(
            1,
            int(config.models.engine_batch_size),
        )
        self._names_validated = False

        if self.backend == "tensorrt":
            weights = Path(
                config.models.weights
            ).expanduser().resolve()
            if weights.suffix != ".engine":
                raise ValueError(
                    "models.backend=tensorrt requires "
                    "models.weights to be a .engine file."
                )
            if not weights.is_file():
                raise FileNotFoundError(
                    f"TensorRT engine not found: {weights}"
                )
            self.model = YOLO(
                str(weights),
                task="segment",
            )
        elif self.backend == "pytorch":
            weights = Path(
                config.models.source_weights
            ).expanduser().resolve()
            if not weights.is_file():
                raise FileNotFoundError(
                    f"YOLOE source weights not found: {weights}"
                )
            model = YOLOE(
                str(weights),
                task="segment",
            )
            text_embeddings = model.get_text_pe(
                list(self.labels)
            )
            model.set_classes(
                list(self.labels),
                text_embeddings,
            )
            model.to(str(self.device))
            self.model = model
        else:
            raise ValueError(
                "models.backend must be 'tensorrt' or "
                f"'pytorch', got {config.models.backend!r}."
            )

        self._warmup()

    def _predict_chunk(
        self,
        images: list[np.ndarray],
    ):
        predict_kwargs = {
            "source": images,
            "imgsz": self.config.models.image_size,
            "conf": self.config.models.confidence,
            "iou": self.config.models.iou,
            "device": str(self.device),
            "retina_masks": self.config.models.retina_masks,
            "verbose": False,
        }

        # TensorRT precision is already fixed when the engine is built.
        # Only the PyTorch fallback needs an inference-precision argument.
        if self.backend == "pytorch":
            predict_kwargs["quantize"] = (
                16 if self.config.models.quantize == 16 else 32
            )

        return self.model.predict(**predict_kwargs)

    def _predict_images(
        self,
        images: list[np.ndarray],
    ) -> list:
        """Run fixed-batch TensorRT while supporting any camera count.

        The final incomplete batch is padded by repeating its last image. Padded
        results are discarded. Normal two-camera operation performs one engine
        invocation.
        """

        if not images:
            return []

        results: list = []
        batch_size = (
            self.engine_batch_size
            if self.backend == "tensorrt"
            else max(1, len(images))
        )
        for begin in range(0, len(images), batch_size):
            chunk = images[begin : begin + batch_size]
            valid_count = len(chunk)
            if (
                self.backend == "tensorrt"
                and valid_count < batch_size
            ):
                chunk = chunk + [chunk[-1]] * (
                    batch_size - valid_count
                )
            chunk_results = self._predict_chunk(chunk)
            results.extend(chunk_results[:valid_count])
        return results

    def _validate_names(self, results: list) -> None:
        if self._names_validated or not results:
            return
        engine_names = _model_names(results[0])
        if engine_names != self.labels:
            raise RuntimeError(
                "YOLOE engine vocabulary does not match "
                "models.label_file.\n"
                f"engine: {engine_names}\n"
                f"file:   {self.labels}\n"
                "Re-export the TensorRT engine using the same "
                "object_labels.txt."
            )
        self._names_validated = True

    def _warmup(self) -> None:
        warmup_frames = max(
            0,
            int(self.config.models.warmup_frames),
        )
        if warmup_frames == 0:
            return

        dummy = np.zeros(
            (
                self.config.models.image_size,
                self.config.models.image_size,
                3,
            ),
            dtype=np.uint8,
        )
        batch = [
            dummy
            for _ in range(
                self.engine_batch_size
                if self.backend == "tensorrt"
                else 1
            )
        ]
        with torch.inference_mode():
            last_results = []
            for _ in range(warmup_frames):
                last_results = self._predict_chunk(batch)
        self._validate_names(last_results)

    def process(
        self,
        frame: MultiCameraFrame,
        profiler: "CycleProfiler | None" = None,
    ) -> dict[str, PerViewResult]:
        ordered_names = [
            name
            for name in self.config.ros.camera_names
            if name in frame.cameras
        ]
        if not ordered_names:
            raise ValueError(
                "MultiCameraFrame contains no configured cameras."
            )

        if profiler is not None:
            profiler.start("step1_geometry_h2d")
        cameras_gpu = {
            name: _frame_to_gpu(
                frame.cameras[name],
                self.device,
            )
            for name in ordered_names
        }
        if profiler is not None:
            profiler.stop("step1_geometry_h2d")

        # ROS publishes rgb8, but Ultralytics assumes that NumPy HWC
        # images are BGR and converts them internally to RGB.
        rgb_images = [
            _rgb8_to_ultralytics_bgr(
                frame.cameras[name].rgb
            )
            for name in ordered_names
        ]
        
        if profiler is not None:
            profiler.start("step1_yoloe_tensorrt")
        with torch.inference_mode():
            results = self._predict_images(rgb_images)
        if profiler is not None:
            profiler.stop("step1_yoloe_tensorrt")
        self._validate_names(results)

        if profiler is not None:
            profiler.start("step1_mask_geometry")

        output: dict[str, PerViewResult] = {}
        for camera_name, result in zip(
            ordered_names,
            results,
            strict=True,
        ):
            camera = cameras_gpu[camera_name]
            height, width = camera.depth.shape
            instances: list[ViewInstance] = []
            exclusion = torch.zeros(
                (height, width),
                device=self.device,
                dtype=torch.bool,
            )

            if (
                result.masks is not None
                and result.boxes is not None
                and len(result.boxes) > 0
            ):
                masks = result.masks.data.to(
                    self.device,
                    non_blocking=True,
                )
                if masks.shape[-2:] != (height, width):
                    masks = F.interpolate(
                        masks[:, None].float(),
                        size=(height, width),
                        mode="bilinear",
                        align_corners=False,
                    )[:, 0]
                masks = masks > 0.5

                eroded_masks = _binary_erode(
                    masks,
                    self.config.masks.object_erosion_pixels,
                )
                exclusion = _binary_dilate(
                    masks,
                    self.config.masks.background_dilation_pixels,
                ).any(dim=0)

                # One small D2H transfer per camera. Dense masks and point clouds
                # remain on the GPU.
                compact = torch.cat(
                    (
                        result.boxes.xyxy,
                        result.boxes.conf[:, None],
                        result.boxes.cls[:, None],
                    ),
                    dim=1,
                ).detach().float().cpu().numpy()

                for local_id in range(masks.shape[0]):
                    points = _masked_points_world(
                        camera,
                        eroded_masks[local_id],
                        self.config.depth.min_m,
                        self.config.depth.max_m,
                    )
                    if points.shape[0] == 0:
                        continue

                    x0, y0, x1, y1, confidence, class_value = (
                        compact[local_id]
                    )
                    class_id = int(class_value)
                    if not 0 <= class_id < len(self.labels):
                        raise RuntimeError(
                            f"YOLOE returned class_id={class_id}, "
                            f"but label file has {len(self.labels)} labels."
                        )

                    instances.append(
                        ViewInstance(
                            camera_name=camera_name,
                            local_instance_id=local_id,
                            class_id=class_id,
                            class_name=self.labels[class_id],
                            class_confidence=float(confidence),
                            bbox_xyxy=(
                                int(round(x0)),
                                int(round(y0)),
                                int(round(x1)),
                                int(round(y1)),
                            ),
                            mask_original=masks[local_id],
                            mask_eroded=eroded_masks[local_id],
                            pcd_world=points,
                            centroid_world=points.median(
                                dim=0
                            ).values,
                            aabb_min_world=points.amin(dim=0),
                            aabb_max_world=points.amax(dim=0),
                            reprojection_points_world=(
                                _uniform_points(
                                    points,
                                    self.config.multiview.reprojection_points,
                                )
                            ),
                        )
                    )

            background_points = _masked_points_world(
                camera,
                ~exclusion,
                self.config.depth.min_m,
                self.config.depth.max_m,
            )
            output[camera_name] = PerViewResult(
                camera=camera,
                instances=instances,
                background_pcd_world=background_points,
            )

        if profiler is not None:
            profiler.stop("step1_mask_geometry")
        return output
