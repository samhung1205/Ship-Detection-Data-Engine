"""
Tests for the transitional GT document container.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.document import AnnotationDocument  # noqa: E402


def test_annotation_document_clear_keeps_list_identity() -> None:
    doc = AnnotationDocument(
        data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_data=[["naval", 10, 20, 30, 40]],
        box_attributes=[{"size_tag": "small"}],
    )
    data_ref = doc.data
    real_ref = doc.real_data
    attrs_ref = doc.box_attributes

    doc.clear()

    assert doc.total_boxes == 0
    assert data_ref is doc.data
    assert real_ref is doc.real_data
    assert attrs_ref is doc.box_attributes
    assert doc.data == []
    assert doc.real_data == []
    assert doc.box_attributes == []


def test_annotation_document_append_box_creates_default_attributes() -> None:
    doc = AnnotationDocument()

    state = doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )

    assert doc.total_boxes == 1
    assert state.real_row == ["naval", 10, 20, 30, 40]
    assert doc.box_attributes[0]["size_tag"] == "small"
    assert doc.box_attributes[0]["crowded"] == "false"


def test_annotation_document_set_box_attributes_fills_missing_defaults() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )

    doc.set_box_attributes(0, {"crowded": "true"})

    assert doc.box_attributes[0]["crowded"] == "true"
    assert doc.box_attributes[0]["difficulty_tag"] == "normal"
    assert doc.box_attributes[0]["size_tag"] == "small"


def test_annotation_document_replace_updates_selected_sections_in_place() -> None:
    doc = AnnotationDocument()
    data_ref = doc.data
    real_ref = doc.real_data
    attrs_ref = doc.box_attributes

    doc.replace(
        data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_data=[["naval", 10, 20, 30, 40]],
    )
    doc.replace(box_attributes=[{"size_tag": "medium", "crowded": "false"}])

    assert data_ref is doc.data
    assert real_ref is doc.real_data
    assert attrs_ref is doc.box_attributes
    assert doc.total_boxes == 1
    assert doc.data[0][:5] == ["naval", 1, 2, 3, 4]
    assert doc.real_data[0] == ["naval", 10, 20, 30, 40]
    assert doc.box_attributes[0]["size_tag"] == "medium"


def test_annotation_document_apply_box_attributes_aligns_to_real_rows() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    doc.append_box(
        ["merchant", 5, 6, 7, 8, 100, 100],
        ["merchant", 50, 60, 70, 80],
    )

    doc.apply_box_attributes([{"crowded": "true"}])

    assert doc.box_attributes[0]["crowded"] == "true"
    assert doc.box_attributes[1]["crowded"] == "false"
    assert doc.box_attributes[1]["size_tag"] == "small"


def test_annotation_document_remove_and_insert_box_preserve_alignment() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    doc.append_box(
        ["merchant", 5, 6, 7, 8, 100, 100],
        ["merchant", 50, 60, 70, 80],
    )

    removed = doc.remove_box(0)

    assert doc.total_boxes == 1
    assert doc.real_data[0][0] == "merchant"

    doc.insert_box(0, removed)

    assert doc.total_boxes == 2
    assert [row[0] for row in doc.real_data] == ["naval", "merchant"]
    doc.validate_alignment()


def test_annotation_document_rename_box_updates_both_row_spaces() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )

    old_name = doc.rename_box(0, "merchant")

    assert old_name == "naval"
    assert doc.data[0][0] == "merchant"
    assert doc.real_data[0][0] == "merchant"


def test_annotation_document_replace_box_updates_geometry_and_keeps_attributes() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    doc.set_box_attributes(0, {"crowded": "true", "size_tag": "large"})

    state = doc.box_state(0)
    state.data_row[1:7] = [11, 12, 33, 44, 200, 120]
    state.real_row[1:5] = [110, 120, 330, 440]
    doc.replace_box(0, state)

    assert doc.data[0] == ["naval", 11, 12, 33, 44, 200, 120]
    assert doc.real_data[0] == ["naval", 110, 120, 330, 440]
    assert doc.box_attributes[0]["crowded"] == "true"
    assert doc.box_attributes[0]["size_tag"] == "large"


def test_annotation_document_recalc_size_tag_updates_existing_attrs() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 40, 50],
    )
    doc.set_box_attributes(0, {"crowded": "true", "size_tag": "medium"})

    size_tag = doc.recalc_size_tag(0)

    assert size_tag == "small"
    assert doc.box_attributes[0]["crowded"] == "true"
    assert doc.box_attributes[0]["size_tag"] == "small"


def test_annotation_document_snapshot_restore_round_trip() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    snap = doc.snapshot()

    doc.append_box(
        ["merchant", 5, 6, 7, 8, 100, 100],
        ["merchant", 50, 60, 70, 80],
    )
    doc.restore(snap)

    assert doc.total_boxes == 1
    assert doc.real_data[0] == ["naval", 10, 20, 30, 40]
    assert doc.box_attributes[0]["size_tag"] == "small"


def test_annotation_document_build_metadata_records_uses_current_gt_state() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    doc.set_box_attributes(0, {"crowded": "true"})

    recs = doc.build_metadata_records(
        image_path="/tmp/a.jpg",
        image_width=960,
        image_height=960,
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel"},
    )

    assert recs[0]["class_id"] == 0
    assert recs[0]["super_category"] == "vessel"
    assert recs[0]["crowded"] == "true"


def test_annotation_document_validate_alignment_accepts_aligned_lists() -> None:
    doc = AnnotationDocument(
        data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_data=[["naval", 10, 20, 30, 40]],
        box_attributes=[{"size_tag": "small"}],
    )

    doc.validate_alignment()


def test_annotation_document_validate_alignment_rejects_mismatched_lengths() -> None:
    doc = AnnotationDocument(
        data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_data=[],
        box_attributes=[],
    )

    with pytest.raises(ValueError, match="len\\(data\\)=1 != len\\(real_data\\)=0"):
        doc.validate_alignment()
