"""Tests for error analysis: IoU, matching, error classification, CSV export."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.prediction import PredictionRecord
from sdde.error_analysis import (
    ERROR_FILTER_ALL,
    ERROR_DUPLICATE,
    ERROR_FN,
    ERROR_FP,
    ERROR_LOCALIZATION,
    ERROR_TP,
    ERROR_WRONG_CLASS,
    ErrorCase,
    export_error_cases_csv,
    filter_error_cases,
    gt_attributes_for_case,
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
    fp_case = next(c for c in cases if c.error_type == ERROR_FP)
    assert fp_case.pred_box == (500.0, 500.0, 600.0, 600.0)


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


def test_filter_error_cases_by_type() -> None:
    cases = [
        ErrorCase(error_type=ERROR_TP),
        ErrorCase(error_type=ERROR_FP),
        ErrorCase(error_type=ERROR_FP),
    ]
    filtered = filter_error_cases(cases, error_type=ERROR_FP)
    assert [c.error_type for c in filtered] == [ERROR_FP, ERROR_FP]


def test_filter_error_cases_by_bookmark() -> None:
    cases = [
        ErrorCase(error_type=ERROR_FP, bookmarked=False),
        ErrorCase(error_type=ERROR_FN, bookmarked=True),
        ErrorCase(error_type=ERROR_TP, bookmarked=True),
    ]
    filtered = filter_error_cases(cases, error_type=ERROR_FILTER_ALL, bookmarked_only=True)
    assert [c.error_type for c in filtered] == [ERROR_FN, ERROR_TP]


def test_filter_error_cases_by_gt_attributes() -> None:
    cases = [
        ErrorCase(error_type=ERROR_FN, gt_index=0),
        ErrorCase(error_type=ERROR_TP, gt_index=1),
        ErrorCase(error_type=ERROR_FP, pred_index=0),
    ]
    gt_attrs = [
        {
            "size_tag": "small",
            "scene_tag": "near_shore",
            "difficulty_tag": "hard",
            "crowded": "true",
            "hard_sample": "true",
            "occluded": "true",
            "truncated": "false",
            "blurred": "true",
        },
        {
            "size_tag": "large",
            "scene_tag": "offshore",
            "difficulty_tag": "normal",
            "crowded": "false",
            "hard_sample": "false",
            "occluded": "false",
            "truncated": "false",
            "blurred": "false",
        },
    ]
    filtered = filter_error_cases(
        cases,
        gt_attributes=gt_attrs,
        size_tag="small",
        scene_tag="near_shore",
        difficulty_tag="hard",
        crowded="true",
        hard_sample="true",
        occluded="true",
        truncated="false",
        blurred="true",
    )
    assert [c.error_type for c in filtered] == [ERROR_FN]


def test_gt_attributes_for_case_returns_none_for_fp_only_case() -> None:
    case = ErrorCase(error_type=ERROR_FP, pred_index=0)
    assert gt_attributes_for_case(case, [{"size_tag": "small"}]) is None


def test_gt_attributes_for_case_normalizes_defaults() -> None:
    case = ErrorCase(error_type=ERROR_FN, gt_index=0)
    attrs = gt_attributes_for_case(case, [{"scene_tag": "near_shore"}])
    assert attrs is not None
    assert attrs["scene_tag"] == "near_shore"
    assert attrs["size_tag"] == "medium"
    assert attrs["difficulty_tag"] == "normal"


def test_match_gt_pred_attaches_gt_attributes_to_cases() -> None:
    cases = match_gt_pred(
        [("ship", 0, 0, 10, 10)],
        [_pred("ship", 0, 0, 10, 10)],
        gt_attributes=[{"scene_tag": "near_shore", "crowded": "true"}],
    )
    attrs = gt_attributes_for_case(cases[0], None)
    assert attrs is not None
    assert attrs["scene_tag"] == "near_shore"
    assert attrs["crowded"] == "true"


def test_export_csv() -> None:
    cases = [ErrorCase(error_type=ERROR_FN, gt_class="ship", image_id="img1")]
    text = export_error_cases_csv(cases)
    assert "error_type" in text
    assert "FN" in text
    assert "ship" in text
