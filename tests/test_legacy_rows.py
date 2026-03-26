"""Tests for bridging legacy GUI rows and SDDE annotation models."""

from sdde.legacy_rows import (
    annotations_from_legacy_rows,
    class_mapping_from_object_list,
    legacy_blocks_from_annotations,
)


def test_annotations_from_legacy_rows() -> None:
    rows = [
        ["merchant", 10, 20, 30, 40],
        ["dock", 50, 60, 80, 100],
    ]

    anns = annotations_from_legacy_rows(rows, object_list=["naval", "merchant", "dock"])

    assert len(anns) == 2
    assert anns[0].class_id == 1
    assert anns[0].bbox_px.x1 == 10
    assert anns[1].class_id == 2
    assert anns[1].bbox_px.y2 == 100


def test_legacy_blocks_from_annotations_roundtrip() -> None:
    object_list = ["naval", "merchant", "dock"]
    mapping = class_mapping_from_object_list(object_list)
    rows = [["merchant", 10, 20, 30, 40]]

    anns = annotations_from_legacy_rows(rows, object_list=object_list)
    blocks = legacy_blocks_from_annotations(
        anns,
        class_mapping=mapping,
        canvas_w=640,
        canvas_h=480,
    )

    assert blocks == [
        (
            ["merchant", 10, 20, 30, 40, 640, 480],
            ["merchant", 10, 20, 30, 40],
            "merchant",
        )
    ]
