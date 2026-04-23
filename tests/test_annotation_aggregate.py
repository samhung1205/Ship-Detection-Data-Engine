"""Tests for GT + paste annotation aggregation helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.annotation_aggregate import (
    build_combined_annotation_records,
    combined_box_attributes,
    combined_gt_boxes,
    combined_real_rows,
)


def test_combined_real_rows_preserve_gt_then_paste_order() -> None:
    rows = combined_real_rows(
        [["naval", 1, 2, 3, 4]],
        [["merchant", 5, 6, 7, 8]],
    )

    assert rows == [
        ["naval", 1, 2, 3, 4],
        ["merchant", 5, 6, 7, 8],
    ]


def test_combined_box_attributes_fill_defaults_for_paste_rows() -> None:
    attrs = combined_box_attributes(
        [{"crowded": "true", "size_tag": "small"}],
        [["merchant", 5, 6, 25, 36]],
    )

    assert attrs[0]["crowded"] == "true"
    assert attrs[0]["size_tag"] == "small"
    assert attrs[1]["crowded"] == "false"
    assert attrs[1]["scene_tag"] == "unknown"


def test_combined_gt_boxes_include_gt_and_paste_rows() -> None:
    boxes = combined_gt_boxes(
        [["naval", 1, 2, 3, 4]],
        [["merchant", 5, 6, 7, 8]],
    )

    assert boxes == [
        ("naval", 1.0, 2.0, 3.0, 4.0),
        ("merchant", 5.0, 6.0, 7.0, 8.0),
    ]


def test_build_combined_annotation_records_marks_sources_and_defaults_paste_attrs() -> None:
    recs = build_combined_annotation_records(
        image_path="/tmp/a.jpg",
        image_width=960,
        image_height=960,
        gt_real_data=[["naval", 0.0, 0.0, 10.0, 10.0]],
        gt_box_attributes=[
            {
                "size_tag": "small",
                "crowded": "true",
                "difficulty_tag": "hard",
                "hard_sample": "true",
                "occluded": "true",
                "truncated": "false",
                "blurred": "true",
                "difficult_background": "true",
                "low_contrast": "false",
                "scene_tag": "near_shore",
            }
        ],
        paste_real_data=[["merchant", 50.0, 60.0, 80.0, 90.0]],
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel", 1: "vessel"},
    )

    assert len(recs) == 2
    assert recs[0]["annotation_source"] == "gt"
    assert recs[0]["crowded"] == "true"
    assert recs[1]["annotation_source"] == "paste"
    assert recs[1]["class_name"] == "merchant"
    assert recs[1]["class_id"] == 1
    assert recs[1]["size_tag"] == "small"
    assert recs[1]["crowded"] == "false"
    assert recs[1]["scene_tag"] == "unknown"
