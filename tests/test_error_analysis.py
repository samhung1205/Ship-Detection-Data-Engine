"""Tests for error analysis: IoU, matching, error classification, CSV export."""

import pytest

from sdde.prediction import PredictionRecord
from sdde.error_analysis import (
    ERROR_DUPLICATE,
    ERROR_FN,
    ERROR_FP,
    ERROR_LOCALIZATION,
    ERROR_TP,
    ERROR_WRONG_CLASS,
    ErrorCase,
    export_error_cases_csv,
    iou_xyxy,
    match_gt_pred,
    summarise_error_cases,
)


def _pred(cls: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.9) -> PredictionRecord:
    return PredictionRecord(
        pred_id="t", class_id=0, class_name=cls,
        x1=x1, y1=y1, x2=x2, y2=y2, confidence=conf,
    )


# --- IoU ---


def test_iou_identical() -> None:
    assert iou_xyxy((0, 0, 10, 10), (0, 0, 10, 10)) == pytest.approx(1.0)


def test_iou_no_overlap() -> None:
    assert iou_xyxy((0, 0, 5, 5), (10, 10, 20, 20)) == pytest.approx(0.0)


def test_iou_partial() -> None:
    v = iou_xyxy((0, 0, 10, 10), (5, 5, 15, 15))
    assert 0 < v < 1
    expected = 25.0 / (100 + 100 - 25)
    assert v == pytest.approx(expected)


# --- Matching ---


def test_match_tp() -> None:
    gts = [("ship", 0, 0, 100, 100)]
    preds = [_pred("ship", 0, 0, 100, 100)]
    cases = match_gt_pred(gts, preds, iou_threshold=0.5)
    types = {c.error_type for c in cases}
    assert types == {ERROR_TP}
    assert cases[0].iou == pytest.approx(1.0)


def test_match_wrong_class() -> None:
    gts = [("ship", 0, 0, 100, 100)]
    preds = [_pred("buoy", 0, 0, 100, 100)]
    cases = match_gt_pred(gts, preds, iou_threshold=0.5)
    assert cases[0].error_type == ERROR_WRONG_CLASS


def test_match_fp_fn() -> None:
    gts = [("ship", 0, 0, 10, 10)]
    preds = [_pred("ship", 500, 500, 600, 600)]
    cases = match_gt_pred(gts, preds, iou_threshold=0.5)
    types = {c.error_type for c in cases}
    assert ERROR_FP in types
    assert ERROR_FN in types


def test_match_localization() -> None:
    gts = [("ship", 0, 0, 100, 100)]
    preds = [_pred("ship", 60, 60, 160, 160)]
    cases = match_gt_pred(gts, preds, iou_threshold=0.5, localization_low=0.01)
    types = [c.error_type for c in cases]
    assert ERROR_LOCALIZATION in types


def test_match_duplicate() -> None:
    gts = [("ship", 0, 0, 100, 100)]
    preds = [
        _pred("ship", 0, 0, 100, 100, 0.95),
        _pred("ship", 0, 0, 100, 100, 0.80),
    ]
    cases = match_gt_pred(gts, preds, iou_threshold=0.5)
    types = [c.error_type for c in cases]
    assert ERROR_TP in types
    assert ERROR_DUPLICATE in types


def test_summarise() -> None:
    cases = [
        ErrorCase(error_type=ERROR_TP),
        ErrorCase(error_type=ERROR_TP),
        ErrorCase(error_type=ERROR_FP),
    ]
    s = summarise_error_cases(cases)
    assert s[ERROR_TP] == 2
    assert s[ERROR_FP] == 1


def test_export_csv() -> None:
    cases = [ErrorCase(error_type=ERROR_FN, gt_class="ship", image_id="img1")]
    text = export_error_cases_csv(cases)
    assert "error_type" in text
    assert "FN" in text
    assert "ship" in text
