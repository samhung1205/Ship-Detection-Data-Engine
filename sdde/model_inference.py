"""
Optional YOLO model inference helpers for GUI-side semi-auto annotation.
"""
from __future__ import annotations

import importlib
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .prediction import PredictionRecord, STATUS_PREDICTED


@dataclass
class YoloModelHandle:
    model_path: str
    backend_name: str
    model: Any


def _load_ultralytics_yolo_class():
    try:
        module = importlib.import_module("ultralytics")
        return module.YOLO
    except (ImportError, AttributeError) as e:
        raise RuntimeError(
            "YOLO model inference requires optional dependencies `ultralytics` and `torch`."
        ) from e


def load_yolo_model(path: str | Path) -> YoloModelHandle:
    p = Path(path)
    if not p.exists():
        raise OSError(f"Model file not found: {p}")
    yolo_cls = _load_ultralytics_yolo_class()
    try:
        model = yolo_cls(str(p))
    except Exception as e:  # pragma: no cover - backend-specific failure formatting
        raise RuntimeError(f"Failed to load YOLO model: {e}") from e
    return YoloModelHandle(model_path=str(p), backend_name="ultralytics", model=model)


def run_yolo_model_inference(
    handle: YoloModelHandle,
    *,
    image_path: str | Path,
    object_list: Sequence[str],
    conf_threshold: float = 0.01,
    iou_threshold: float = 0.7,
    max_det: int = 300,
) -> list[PredictionRecord]:
    if not object_list:
        raise ValueError("object_list cannot be empty")

    try:
        results = handle.model.predict(
            source=str(image_path),
            conf=max(0.0, min(1.0, float(conf_threshold))),
            iou=max(0.0, min(1.0, float(iou_threshold))),
            max_det=max(1, int(max_det)),
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - backend-specific failure formatting
        raise RuntimeError(f"YOLO inference failed: {e}") from e

    if not results:
        return []
    result = results[0]
    box_container = _select_box_container(result)
    if box_container is None:
        return []

    xyxy_rows = _tensor_to_list(getattr(box_container, "xyxy", None))
    cls_values = _tensor_to_list(getattr(box_container, "cls", None))
    conf_values = _tensor_to_list(getattr(box_container, "conf", None))
    if not xyxy_rows:
        return []

    predictions: list[PredictionRecord] = []
    for idx, row in enumerate(xyxy_rows):
        if len(row) < 4:
            continue
        class_id = int(float(cls_values[idx])) if idx < len(cls_values) else 0
        if class_id < 0 or class_id >= len(object_list):
            raise IndexError(f"Model predicted class_id {class_id} outside current classes.yaml range")
        confidence = float(conf_values[idx]) if idx < len(conf_values) else 1.0
        predictions.append(
            PredictionRecord(
                pred_id=str(uuid.uuid4())[:12],
                class_id=class_id,
                class_name=str(object_list[class_id]),
                x1=float(row[0]),
                y1=float(row[1]),
                x2=float(row[2]),
                y2=float(row[3]),
                confidence=confidence,
                pred_status=STATUS_PREDICTED,
            )
        )
    return predictions


def _select_box_container(result: Any) -> Any | None:
    boxes = getattr(result, "boxes", None)
    if boxes is not None and getattr(boxes, "xyxy", None) is not None:
        return boxes
    obb = getattr(result, "obb", None)
    if obb is not None and getattr(obb, "xyxy", None) is not None:
        return obb
    return None


def _tensor_to_list(value: Any) -> list:
    if value is None:
        return []
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "tolist"):
        data = value.tolist()
        return list(data) if isinstance(data, (list, tuple)) else [data]
    if isinstance(value, (list, tuple)):
        return list(value)
    return []
