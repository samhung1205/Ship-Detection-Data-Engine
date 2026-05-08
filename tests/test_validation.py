"""Tests for dataset / project validation helpers."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import json

from sdde.validation import (
    build_validation_summary,
    export_validation_issues_csv,
    export_validation_summary_json,
    scan_dataset_validation,
)


def _write_image(path: Path, *, width: int = 40, height: int = 20) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_scan_dataset_validation_reports_missing_and_invalid_sidecars(tmp_path: Path) -> None:
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _write_image(img_a)
    _write_image(img_b)
    (tmp_path / "a.txt").write_text("3 0.5 0.5 1.2 0.5\n", encoding="utf-8")
    pred_root = tmp_path / "predictions"
    pred_root.mkdir()
    (pred_root / "a.txt").write_text("0 1.2 0.5 0.4 0.4 1.5\n", encoding="utf-8")

    result = scan_dataset_validation(
        tmp_path,
        object_list=["naval", "merchant"],
        prediction_root=pred_root,
    )

    assert result.total_images == 2
    assert result.matched_labels == 1
    assert result.matched_predictions == 1
    issue_types = {issue.issue_type for issue in result.issues}
    assert "invalid_label_class_id" in issue_types
    assert "label_size_out_of_range" in issue_types
    assert "label_bbox_out_of_bounds" in issue_types
    assert "prediction_center_out_of_range" in issue_types
    assert "prediction_confidence_out_of_range" in issue_types
    assert "missing_label" in issue_types
    assert "missing_prediction" in issue_types


def test_scan_dataset_validation_supports_recursive_project_prediction_mapping(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    pred_root = tmp_path / "predictions"
    (image_root / "train").mkdir(parents=True)
    (label_root / "train").mkdir(parents=True)
    (pred_root / "train").mkdir(parents=True)

    img = image_root / "train" / "ship.jpg"
    _write_image(img)
    (label_root / "train" / "ship.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")
    (pred_root / "train" / "ship.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    result = scan_dataset_validation(
        image_root,
        object_list=["naval", "merchant"],
        recursive=True,
        image_root=image_root,
        label_root=label_root,
        prediction_root=pred_root,
    )

    assert result.total_images == 1
    assert result.matched_labels == 1
    assert result.matched_predictions == 1
    assert result.total_issues == 0


def test_export_validation_issues_csv_includes_issue_rows(tmp_path: Path) -> None:
    img = tmp_path / "sample.jpg"
    _write_image(img)

    result = scan_dataset_validation(
        tmp_path,
        object_list=["naval"],
    )

    body = export_validation_issues_csv(result)
    assert "issue_type" in body
    assert "missing_label" in body
    assert str(img) in body


def test_validation_summary_reports_clean_and_issue_counts(tmp_path: Path) -> None:
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.jpg"
    _write_image(img_a)
    _write_image(img_b)
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    result = scan_dataset_validation(
        tmp_path,
        object_list=["naval"],
    )

    summary = build_validation_summary(result)
    assert summary["total_images"] == 2
    assert summary["matched_labels"] == 1
    assert summary["clean_images"] == 1
    assert summary["images_with_issues"] == 1
    assert summary["issue_type_counts"]["missing_label"] == 1
    payload = json.loads(export_validation_summary_json(result))
    assert payload["scope_path"] == str(tmp_path)
    assert payload["per_image_issue_counts"][str(img_b.resolve())] == 1
