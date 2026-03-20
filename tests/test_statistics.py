"""Tests for dataset statistics computation and export."""

import json
import pytest

from sdde.statistics import compute_dataset_stats, export_stats_csv, export_stats_json


def _rec(cls: str, x1: float, y1: float, x2: float, y2: float, st: str = "", img: str = "img.jpg"):
    return {
        "image_path": img,
        "class_name": cls,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "size_tag": st or None,
    }


def test_basic_counts() -> None:
    recs = [
        _rec("ship", 0, 0, 100, 100),
        _rec("buoy", 10, 10, 20, 20),
    ]
    s = compute_dataset_stats(recs)
    assert s["total_images"] == 1
    assert s["total_annotations"] == 2


def test_class_distribution() -> None:
    recs = [_rec("ship", 0, 0, 10, 10)] * 3 + [_rec("buoy", 0, 0, 5, 5)]
    s = compute_dataset_stats(recs)
    assert s["class_distribution"]["ship"] == 3
    assert s["class_distribution"]["buoy"] == 1


def test_size_tag_distribution_uses_compute() -> None:
    recs = [
        _rec("a", 0, 0, 10, 10),        # 100 px² → small
        _rec("b", 0, 0, 100, 100),       # 10000 → large
    ]
    s = compute_dataset_stats(recs)
    assert "small" in s["size_tag_distribution"]
    assert "large" in s["size_tag_distribution"]


def test_bbox_numeric_summary() -> None:
    recs = [_rec("x", 0, 0, 10, 20)]
    s = compute_dataset_stats(recs)
    wd = s["bbox_width_distribution"]
    hd = s["bbox_height_distribution"]
    assert wd["count"] == 1
    assert wd["min"] == pytest.approx(10.0)
    assert hd["min"] == pytest.approx(20.0)


def test_class_x_size() -> None:
    recs = [
        _rec("ship", 0, 0, 10, 10),        # small
        _rec("ship", 0, 0, 200, 200),       # large
    ]
    s = compute_dataset_stats(recs)
    cxs = s["class_x_size_distribution"]
    assert "ship" in cxs
    assert cxs["ship"].get("small", 0) >= 1 or cxs["ship"].get("large", 0) >= 1


def test_empty_records() -> None:
    s = compute_dataset_stats([])
    assert s["total_annotations"] == 0
    assert s["total_images"] == 0


def test_export_json() -> None:
    recs = [_rec("ship", 0, 0, 50, 50)]
    s = compute_dataset_stats(recs)
    text = export_stats_json(s)
    data = json.loads(text)
    assert data["total_annotations"] == 1


def test_export_csv() -> None:
    recs = [_rec("ship", 0, 0, 50, 50)]
    s = compute_dataset_stats(recs)
    text = export_stats_csv(s)
    assert "total_images" in text
    assert "class_distribution" in text
