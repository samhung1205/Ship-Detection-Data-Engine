"""
Helpers for folder-sidecar prediction review workflows.
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from .prediction import PredictionRecord, parse_predictions_yolo_txt


def prediction_sidecar_path(
    image_path: str | Path,
    *,
    prediction_root: str | Path,
) -> Path:
    image = Path(image_path)
    return Path(prediction_root) / f"{image.stem}.txt"


def has_prediction_sidecar(
    image_path: str | Path,
    *,
    prediction_root: str | Path,
) -> bool:
    return prediction_sidecar_path(image_path, prediction_root=prediction_root).is_file()


def load_prediction_sidecar(
    image_path: str | Path,
    *,
    prediction_root: str | Path,
    object_list: Sequence[str],
    image_w: int,
    image_h: int,
) -> list[PredictionRecord]:
    path = prediction_sidecar_path(image_path, prediction_root=prediction_root)
    body = path.read_text(encoding="utf-8")
    return parse_predictions_yolo_txt(
        body,
        object_list=object_list,
        image_w=image_w,
        image_h=image_h,
    )
