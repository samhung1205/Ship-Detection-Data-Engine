"""Tests for folder-level error analysis scanning."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.dataset_scan import ImageAnnotationBundle
from sdde.error_analysis import ERROR_FN, ERROR_TP
from sdde.error_analysis_scan import scan_folder_error_cases
from sdde.prediction import PredictionRecord


def _write_image(path: Path, *, width: int = 40, height: int = 20) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_scan_folder_error_cases_reads_label_and_prediction_sidecars(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    _write_image(img)
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    pred_root = tmp_path / "preds"
    pred_root.mkdir()
    (pred_root / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    result = scan_folder_error_cases(
        tmp_path,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
    )

    assert result.total_images == 1
    assert result.analyzed_images == 1
    assert len(result.cases) == 1
    assert result.cases[0].error_type == ERROR_TP


def test_scan_folder_error_cases_uses_current_image_overrides(tmp_path: Path) -> None:
    img = tmp_path / "a.jpg"
    _write_image(img)
    pred_root = tmp_path / "preds"
    pred_root.mkdir()

    bundle = ImageAnnotationBundle(
        image_path=str(img),
        image_width=40,
        image_height=20,
        records=(
            {
                "image_path": str(img),
                "class_name": "naval",
                "x1": 0.0,
                "y1": 0.0,
                "x2": 10.0,
                "y2": 10.0,
                "size_tag": "small",
                "crowded": "true",
                "difficulty_tag": "hard",
                "hard_sample": "false",
                "occluded": "false",
                "truncated": "false",
                "blurred": "false",
                "difficult_background": "false",
                "low_contrast": "false",
                "scene_tag": "near_shore",
            },
        ),
        gt_boxes=(("naval", 0.0, 0.0, 10.0, 10.0),),
        gt_attributes=({"size_tag": "small", "crowded": "true", "scene_tag": "near_shore"},),
        has_label=True,
    )
    preds = [
        PredictionRecord(
            pred_id="p1",
            class_id=0,
            class_name="naval",
            x1=100,
            y1=100,
            x2=110,
            y2=110,
            confidence=0.95,
        )
    ]

    result = scan_folder_error_cases(
        tmp_path,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
        current_image_path=img,
        current_image_gt_bundle=bundle,
        current_image_predictions=preds,
    )

    assert result.analyzed_images == 1
    error_types = {case.error_type for case in result.cases}
    assert ERROR_FN in error_types
    fn_case = next(case for case in result.cases if case.error_type == ERROR_FN)
    assert fn_case.gt_attrs is not None
    assert fn_case.gt_attrs["crowded"] == "true"
