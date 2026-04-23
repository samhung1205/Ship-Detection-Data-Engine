"""Tests for dataset statistics computation and export."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.statistics import compute_dataset_stats, export_stats_csv, export_stats_json


def _rec(
    cls: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    st: str = "",
    img: str = "img.jpg",
    *,
    scene: str = "unknown",
    rotation_deg: float | None = None,
):
    rec = {
        "image_path": img,
        "class_name": cls,
        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
        "size_tag": st or None,
        "scene_tag": scene,
    }
    if rotation_deg is not None:
        rec["rotation_deg"] = rotation_deg
    return rec


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


def test_avg_annotations_per_image_and_scene_distribution() -> None:
    recs = [
        _rec("ship", 0, 0, 20, 20, img="a.jpg", scene="near_shore"),
        _rec("ship", 0, 0, 30, 30, img="a.jpg", scene="near_shore"),
        _rec("buoy", 0, 0, 15, 15, img="b.jpg", scene="offshore"),
    ]
    s = compute_dataset_stats(recs)
    assert s["total_images"] == 2
    assert s["avg_annotations_per_image"] == pytest.approx(1.5)
    assert s["scene_tag_distribution"]["near_shore"] == 2
    assert s["scene_tag_distribution"]["offshore"] == 1
    assert s["scene_tag_ratio"]["near_shore"] == pytest.approx(66.67)
    assert s["scene_tag_ratio"]["offshore"] == pytest.approx(33.33)


def test_total_image_override_tracks_unlabeled_images() -> None:
    recs = [
        _rec("ship", 0, 0, 20, 20, img="a.jpg"),
        _rec("ship", 0, 0, 20, 20, img="b.jpg"),
    ]
    s = compute_dataset_stats(
        recs,
        total_images_override=4,
        labeled_images_override=2,
    )
    assert s["total_images"] == 4
    assert s["labeled_images"] == 2
    assert s["unlabeled_images"] == 2
    assert s["avg_annotations_per_image"] == pytest.approx(0.5)


def test_rotation_angle_distribution_uses_optional_field() -> None:
    recs = [
        _rec("ship", 0, 0, 20, 20, rotation_deg=10.0),
        _rec("ship", 0, 0, 20, 20, rotation_deg=20.0),
    ]
    s = compute_dataset_stats(recs)
    rot = s["rotation_angle_distribution"]
    assert rot["count"] == 2
    assert rot["min"] == pytest.approx(10.0)
    assert rot["max"] == pytest.approx(20.0)
    assert rot["mean"] == pytest.approx(15.0)


def test_export_json() -> None:
    recs = [_rec("ship", 0, 0, 50, 50)]
    s = compute_dataset_stats(recs)
    text = export_stats_json(s)
    data = json.loads(text)
    assert data["total_annotations"] == 1


def test_export_csv() -> None:
    recs = [_rec("ship", 0, 0, 50, 50, scene="near_shore", rotation_deg=12.5)]
    s = compute_dataset_stats(recs)
    text = export_stats_csv(s)
    assert "avg_annotations_per_image" in text
    assert "labeled_images" in text
    assert "class_distribution" in text
    assert "scene_tag_distribution" in text
    assert "rotation_angle_distribution" in text
