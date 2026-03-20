"""Tests for YOLO prediction parsing (optional confidence field)."""

import pytest

from sdde.prediction import PredictionRecord, parse_predictions_yolo_txt


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
