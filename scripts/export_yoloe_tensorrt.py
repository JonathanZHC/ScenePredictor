#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys

from ultralytics import YOLOE


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scene_pred_pipeline.labels import load_object_labels


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a fixed-vocabulary YOLOE segmentation model "
            "to a fixed-shape TensorRT engine."
        )
    )
    parser.add_argument(
        "--weights",
        default="/workspace/weights/yoloe-26s-seg.pt",
        help="Source YOLOE segmentation .pt file.",
    )
    parser.add_argument(
        "--labels",
        default="/workspace/configs/object_labels.txt",
        help="One text label per line; order defines engine class IDs.",
    )
    parser.add_argument(
        "--output",
        default="/workspace/weights/yoloe-26s-seg.engine",
        help="Destination TensorRT engine path.",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Square YOLO input size.",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=2,
        help="Fixed TensorRT batch size.",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="CUDA device passed to Ultralytics export.",
    )
    parser.add_argument(
        "--workspace",
        type=float,
        default=4.0,
        help="Maximum TensorRT builder workspace in GiB.",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="Export FP32 instead of the default FP16 engine.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    weights = Path(args.weights).expanduser().resolve()
    labels_path = Path(args.labels).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()

    if not weights.is_file():
        raise FileNotFoundError(
            f"YOLOE source weights do not exist: {weights}"
        )
    labels = load_object_labels(labels_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    model = YOLOE(str(weights), task="segment")
    text_embeddings = model.get_text_pe(list(labels))
    model.set_classes(list(labels), text_embeddings)

    exported = model.export(
        format="engine",
        imgsz=args.imgsz,
        batch=args.batch,
        dynamic=False,
        quantize=None if args.fp32 else 16,
        simplify=True,
        workspace=args.workspace,
        device=args.device,
    )
    exported_path = Path(str(exported)).expanduser().resolve()
    if not exported_path.is_file():
        raise RuntimeError(
            "Ultralytics reported a TensorRT export path that does not "
            f"exist: {exported_path}"
        )

    if exported_path != output:
        shutil.copy2(exported_path, output)

    # Keep a sidecar snapshot so it is obvious which vocabulary was embedded.
    sidecar = output.with_suffix(output.suffix + ".labels.txt")
    shutil.copy2(labels_path, sidecar)

    precision = "FP32" if args.fp32 else "FP16"
    print("TensorRT export complete")
    print(f"  engine:    {output}")
    print(f"  labels:    {sidecar}")
    print(f"  classes:   {len(labels)}")
    print(f"  batch:     {args.batch}")
    print(f"  image:     {args.imgsz}x{args.imgsz}")
    print(f"  precision: {precision}")


if __name__ == "__main__":
    main()
