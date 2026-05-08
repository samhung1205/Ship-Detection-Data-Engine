"""Tests for folder / project prediction review summary helpers."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.prediction import PredictionRecord
from sdde.prediction_review import initial_prediction_review_state, update_prediction_review_state
from sdde.prediction_review_report import (
    build_prediction_review_report_summary,
    export_prediction_review_report_csv,
    scan_prediction_review_report,
)
from sdde.prediction_review_store import save_prediction_review_session


def _write_image(path: Path, *, width: int = 40, height: int = 20) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def _pred(pred_id: str) -> PredictionRecord:
    return PredictionRecord(pred_id, 0, "naval", 0, 0, 10, 10, 0.9)


def test_scan_prediction_review_report_reads_saved_folder_state(tmp_path: Path) -> None:
    img_folder = tmp_path / "images"
    pred_root = tmp_path / "predictions"
    img_folder.mkdir()
    pred_root.mkdir()
    img_a = img_folder / "a.jpg"
    img_b = img_folder / "b.jpg"
    _write_image(img_a)
    _write_image(img_b)
    (pred_root / "a.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")
    (pred_root / "b.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")

    save_prediction_review_session(
        image_folder=img_folder,
        prediction_folder=pred_root,
        review_root=tmp_path,
        states={
            str(img_a.resolve()): update_prediction_review_state(
                initial_prediction_review_state([_pred("p1")]),
                accepted_delta=1,
                remaining_predictions=[],
            )
        },
    )

    report = scan_prediction_review_report(
        img_folder,
        prediction_root=pred_root,
        review_root=tmp_path,
    )

    assert report.total_images == 2
    assert report.images_with_predictions == 2
    assert report.reviewed_images == 1
    assert report.pending_images == 1
    assert report.total_accepted_predictions == 1
    csv_body = export_prediction_review_report_csv(report)
    assert "reviewed" in csv_body
    assert str(img_b.resolve()) in csv_body


def test_scan_prediction_review_report_supports_recursive_project_mapping(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    pred_root = tmp_path / "predictions"
    (image_root / "train").mkdir(parents=True)
    (pred_root / "train").mkdir(parents=True)
    img = image_root / "train" / "ship.jpg"
    _write_image(img)
    (pred_root / "train" / "ship.txt").write_text(
        "0 0.5 0.5 0.4 0.4 0.8\n0 0.2 0.2 0.1 0.1 0.7\n",
        encoding="utf-8",
    )

    report = scan_prediction_review_report(
        image_root,
        prediction_root=pred_root,
        review_root=tmp_path,
        recursive=True,
        image_root=image_root,
    )

    assert report.total_images == 1
    assert report.images_with_predictions == 1
    assert report.pending_images == 1
    assert report.total_original_predictions == 2
    summary = build_prediction_review_report_summary(report)
    assert summary["pending_images"] == 1


def test_scan_prediction_review_report_prefers_current_state_over_saved_state(tmp_path: Path) -> None:
    img_folder = tmp_path / "images"
    pred_root = tmp_path / "predictions"
    img_folder.mkdir()
    pred_root.mkdir()
    img = img_folder / "a.jpg"
    _write_image(img)
    (pred_root / "a.txt").write_text("0 0.5 0.5 0.4 0.4 0.8\n", encoding="utf-8")

    save_prediction_review_session(
        image_folder=img_folder,
        prediction_folder=pred_root,
        review_root=tmp_path,
        states={
            str(img.resolve()): update_prediction_review_state(
                initial_prediction_review_state([_pred("p1")]),
                rejected_delta=1,
                remaining_predictions=[],
            )
        },
    )

    current_state = update_prediction_review_state(
        initial_prediction_review_state([_pred("p1")]),
        accepted_delta=1,
        remaining_predictions=[],
    )
    report = scan_prediction_review_report(
        img_folder,
        prediction_root=pred_root,
        review_root=tmp_path,
        current_states={str(img.resolve()): current_state},
    )

    assert report.reviewed_images == 1
    assert report.total_accepted_predictions == 1
    assert report.total_rejected_predictions == 0
