from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import json
from typing import Any, Sequence

import torch


@dataclass(frozen=True)
class YoloDetectionRecord:
    """One raw 2D YOLO detection from one camera view."""

    local_id: int
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


def _records_from_result(
    result: Any,
    labels: Sequence[str],
) -> list[YoloDetectionRecord]:
    """Convert one Ultralytics result with one compact GPU->CPU transfer."""

    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    compact = torch.cat(
        (
            boxes.xyxy,
            boxes.conf[:, None],
            boxes.cls[:, None],
        ),
        dim=1,
    ).detach().float().cpu().numpy()

    records: list[YoloDetectionRecord] = []
    for local_id, row in enumerate(compact):
        x0, y0, x1, y1, confidence, class_value = row
        class_id = int(class_value)
        if not 0 <= class_id < len(labels):
            raise RuntimeError(
                f"YOLO returned class_id={class_id}, but the configured "
                f"label list contains {len(labels)} classes."
            )

        records.append(
            YoloDetectionRecord(
                local_id=local_id,
                class_id=class_id,
                class_name=str(labels[class_id]),
                confidence=float(confidence),
                bbox_xyxy=(
                    int(round(float(x0))),
                    int(round(float(y0))),
                    int(round(float(x1))),
                    int(round(float(y1))),
                ),
            )
        )

    records.sort(
        key=lambda item: (
            -item.confidence,
            item.class_id,
            item.local_id,
        )
    )
    return records


class PerViewYoloPrinter:
    """Print raw YOLO detections independently for every configured camera."""

    def __init__(
        self,
        *,
        labels: Sequence[str],
        every_n_frames: int = 1,
        output_format: str = "text",
        print_empty: bool = True,
        minimum_confidence: float = 0.0,
    ) -> None:
        if every_n_frames <= 0:
            raise ValueError("every_n_frames must be positive.")
        if output_format not in {"text", "json"}:
            raise ValueError(
                "output_format must be 'text' or 'json'."
            )
        if not 0.0 <= minimum_confidence <= 1.0:
            raise ValueError(
                "minimum_confidence must be in [0, 1]."
            )

        self.labels = tuple(str(value) for value in labels)
        self.every_n_frames = int(every_n_frames)
        self.output_format = output_format
        self.print_empty = bool(print_empty)
        self.minimum_confidence = float(minimum_confidence)

    def emit(
        self,
        *,
        frame_index: int,
        stamp_ns: int,
        camera_names: Sequence[str],
        results: Sequence[Any],
    ) -> None:
        """Print one synchronized multi-camera YOLO timestep."""

        if frame_index % self.every_n_frames != 0:
            return

        if len(results) != len(camera_names):
            raise RuntimeError(
                "YOLO result count does not match camera count: "
                f"results={len(results)}, cameras={len(camera_names)}."
            )

        per_camera: dict[str, list[YoloDetectionRecord]] = {}
        for camera_name, result in zip(
            camera_names,
            results,
            strict=True,
        ):
            records = [
                item
                for item in _records_from_result(
                    result,
                    self.labels,
                )
                if item.confidence >= self.minimum_confidence
            ]
            per_camera[str(camera_name)] = records

        if (
            not self.print_empty
            and not any(per_camera.values())
        ):
            return

        if self.output_format == "json":
            self._emit_json(
                frame_index=frame_index,
                stamp_ns=stamp_ns,
                per_camera=per_camera,
            )
        else:
            self._emit_text(
                frame_index=frame_index,
                stamp_ns=stamp_ns,
                per_camera=per_camera,
            )

    @staticmethod
    def _emit_text(
        *,
        frame_index: int,
        stamp_ns: int,
        per_camera: dict[
            str,
            list[YoloDetectionRecord],
        ],
    ) -> None:
        lines = [
            (
                f"[YOLOE] frame={frame_index:06d} "
                f"stamp_ns={stamp_ns}"
            )
        ]

        for camera_name, records in per_camera.items():
            counts = Counter(
                item.class_name
                for item in records
            )
            summary = (
                ", ".join(
                    f"{name}={count}"
                    for name, count in sorted(counts.items())
                )
                if counts
                else "none"
            )
            lines.append(
                f"  {camera_name}: "
                f"{len(records)} detection(s) [{summary}]"
            )

            for item in records:
                x0, y0, x1, y1 = item.bbox_xyxy
                lines.append(
                    "    "
                    f"#{item.local_id:02d} "
                    f"{item.class_name:<14} "
                    f"conf={item.confidence:.3f} "
                    f"bbox=({x0}, {y0}, {x1}, {y1})"
                )

        print("\n".join(lines), flush=True)

    @staticmethod
    def _emit_json(
        *,
        frame_index: int,
        stamp_ns: int,
        per_camera: dict[
            str,
            list[YoloDetectionRecord],
        ],
    ) -> None:
        payload = {
            "frame_index": int(frame_index),
            "stamp_ns": int(stamp_ns),
            "cameras": {
                camera_name: [
                    asdict(item)
                    for item in records
                ]
                for camera_name, records
                in per_camera.items()
            },
        }
        print(
            json.dumps(
                payload,
                separators=(",", ":"),
                sort_keys=False,
            ),
            flush=True,
        )
