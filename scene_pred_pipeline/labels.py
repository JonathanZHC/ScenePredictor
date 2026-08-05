from __future__ import annotations

from pathlib import Path


def load_object_labels(path: str | Path) -> tuple[str, ...]:
    """Read one object label per line.

    Empty lines and lines starting with '#' are ignored. Labels are kept in
    file order because the TensorRT engine class indices use this exact order.
    """

    label_path = Path(path).expanduser().resolve()
    if not label_path.is_file():
        raise FileNotFoundError(
            f"Object label file does not exist: {label_path}"
        )

    labels: list[str] = []
    with label_path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            label = raw_line.strip()
            if not label or label.startswith("#"):
                continue
            if label in labels:
                raise ValueError(
                    f"Duplicate label {label!r} at "
                    f"{label_path}:{line_number}"
                )
            labels.append(label)

    if not labels:
        raise ValueError(
            f"Object label file contains no labels: {label_path}"
        )
    return tuple(labels)
