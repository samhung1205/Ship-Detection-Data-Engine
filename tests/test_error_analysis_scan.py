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
    assert result.labeled_images == 1
    assert result.prediction_images == 1
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
    assert result.labeled_images == 1
    assert result.prediction_images == 1
    error_types = {case.error_type for case in result.cases}
    assert ERROR_FN in error_types
    fn_case = next(case for case in result.cases if case.error_type == ERROR_FN)
    assert fn_case.gt_attrs is not None
    assert fn_case.gt_attrs["crowded"] == "true"


def test_scan_folder_error_cases_infers_sibling_labels_and_tracks_prediction_matches(tmp_path: Path) -> None:
    image_folder = tmp_path / "images"
    label_folder = tmp_path / "labels"
    pred_root = tmp_path / "predictions"
    image_folder.mkdir()
    label_folder.mkdir()
    pred_root.mkdir()

    img_a = image_folder / "a.jpg"
    img_b = image_folder / "b.jpg"
    _write_image(img_a)
    _write_image(img_b)
    (label_folder / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (label_folder / "b.txt").write_text("1 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    (pred_root / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    result = scan_folder_error_cases(
        image_folder,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
        image_root=tmp_path / "dataset",
        label_root=tmp_path / "dataset",
    )

    assert result.total_images == 2
    assert result.analyzed_images == 2
    assert result.labeled_images == 2
    assert result.prediction_images == 1
    assert {Path(path).name for path in result.prediction_image_paths} == {"a.jpg"}
    error_types = {case.error_type for case in result.cases}
    assert ERROR_TP in error_types
    assert ERROR_FN in error_types


def test_scan_folder_error_cases_supports_recursive_project_prediction_mapping(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    pred_root = tmp_path / "predictions"
    (image_root / "train").mkdir(parents=True)
    (image_root / "val").mkdir(parents=True)
    (label_root / "train").mkdir(parents=True)
    (label_root / "val").mkdir(parents=True)
    (pred_root / "train").mkdir(parents=True)

    img_a = image_root / "train" / "a.jpg"
    img_b = image_root / "val" / "b.jpg"
    _write_image(img_a)
    _write_image(img_b)
    (label_root / "train" / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (label_root / "val" / "b.txt").write_text("1 0.5 0.5 0.25 0.5\n", encoding="utf-8")
    (pred_root / "train" / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    result = scan_folder_error_cases(
        image_root,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
        recursive=True,
        image_root=image_root,
        label_root=label_root,
    )

    assert result.total_images == 2
    assert result.labeled_images == 2
    assert result.prediction_images == 1
    assert {Path(path).name for path in result.prediction_image_paths} == {"a.jpg"}


def test_scan_folder_error_cases_falls_back_to_flat_prediction_folder(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    pred_root = tmp_path / "predictions"
    image_root.mkdir()
    pred_root.mkdir()

    img = image_root / "a.jpg"
    _write_image(img)
    (pred_root / "a.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    result = scan_folder_error_cases(
        image_root,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
        image_root=tmp_path,
    )

    assert result.total_images == 1
    assert result.prediction_images == 1
    assert {Path(path).name for path in result.prediction_image_paths} == {"a.jpg"}
    assert {case.error_type for case in result.cases} == {"FP"}
