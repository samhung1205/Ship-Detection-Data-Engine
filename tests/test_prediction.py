"""Tests for YOLO prediction parsing and confidence filtering."""

import pytest

from sdde.prediction import (
    PredictionRecord,
    filter_predictions_by_confidence,
    parse_predictions_yolo_txt,
    rename_prediction_class,
    update_prediction_geometry_from_canvas_rect,
)


def test_parse_predictions_yolo_txt_with_confidence() -> None:
    names = ["a", "b", "c"]
    content = "0 0.5 0.5 0.2 0.4 0.87\n"
    preds = parse_predictions_yolo_txt(
        content, object_list=names, image_w=1000, image_h=500
    )
    assert len(preds) == 1
    p = preds[0]
    assert p.class_id == 0
    assert p.class_name == "a"
    assert p.confidence == pytest.approx(0.87)
    assert p.x1 == pytest.approx(400.0)
    assert p.y1 == pytest.approx(150.0)
    assert p.x2 == pytest.approx(600.0)
    assert p.y2 == pytest.approx(350.0)


def test_parse_predictions_yolo_txt_default_confidence() -> None:
    names = ["x"]
    content = "# comment\n0 0.1 0.1 0.02 0.02\n"
    preds = parse_predictions_yolo_txt(
        content, object_list=names, image_w=100, image_h=100
    )
    assert len(preds) == 1
    assert preds[0].confidence == pytest.approx(1.0)


def test_prediction_record_bad_class_id() -> None:
    with pytest.raises(IndexError):
        PredictionRecord.from_yolo_line(
            ["99", "0.5", "0.5", "1", "1"],
            object_list=["a"],
            image_w=10,
            image_h=10,
        )


def test_filter_predictions_by_confidence_keeps_expected_rows() -> None:
    preds = [
        PredictionRecord("p1", 0, "a", 0, 0, 10, 10, 0.24),
        PredictionRecord("p2", 0, "a", 0, 0, 10, 10, 0.25),
        PredictionRecord("p3", 0, "a", 0, 0, 10, 10, 0.91),
    ]
    kept = filter_predictions_by_confidence(preds, min_confidence=0.25)
    assert [p.pred_id for p in kept] == ["p2", "p3"]


def test_filter_predictions_by_confidence_clamps_threshold() -> None:
    preds = [PredictionRecord("p1", 0, "a", 0, 0, 10, 10, 0.60)]
    assert filter_predictions_by_confidence(preds, min_confidence=-1.0) == preds
    assert filter_predictions_by_confidence(preds, min_confidence=2.0) == []


def test_rename_prediction_class_updates_class_and_status() -> None:
    preds = [PredictionRecord("p1", 0, "a", 0, 0, 10, 10, 0.60)]

    ok = rename_prediction_class(
        preds,
        0,
        new_class_name="b",
        object_list=["a", "b", "c"],
    )

    assert ok is True
    assert preds[0].class_id == 1
    assert preds[0].class_name == "b"
    assert preds[0].pred_status == "edited"


def test_update_prediction_geometry_from_canvas_rect_updates_origin_space() -> None:
    preds = [PredictionRecord("p1", 0, "a", 0, 0, 10, 10, 0.60)]

    ok = update_prediction_geometry_from_canvas_rect(
        preds,
        0,
        x1=20,
        y1=30,
        x2=60,
        y2=90,
        canvas_width=100,
        canvas_height=100,
        origin_width=200,
        origin_height=300,
    )

    assert ok is True
    assert preds[0].x1 == pytest.approx(40.0)
    assert preds[0].y1 == pytest.approx(90.0)
    assert preds[0].x2 == pytest.approx(120.0)
    assert preds[0].y2 == pytest.approx(270.0)
    assert preds[0].pred_status == "edited"
