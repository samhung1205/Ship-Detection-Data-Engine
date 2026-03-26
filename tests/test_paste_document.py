"""
Tests for the transitional committed paste document container.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.paste_document import PasteDocument  # noqa: E402


class _NoDeepcopy:
    def __deepcopy__(self, memo):
        raise TypeError("no deepcopy")


def test_paste_document_clear_keeps_list_identity() -> None:
    doc = PasteDocument(
        pimg_data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_pimg_data=[["naval", 10, 20, 30, 40]],
        paste_images=[["rgba", 0.1, 0.2, 0.3, 0.4]],
        paste_records=[SimpleNamespace(class_name="naval")],
    )
    pimg_ref = doc.pimg_data
    real_ref = doc.real_pimg_data
    images_ref = doc.paste_images
    records_ref = doc.paste_records

    doc.clear()

    assert doc.total_pastes == 0
    assert pimg_ref is doc.pimg_data
    assert real_ref is doc.real_pimg_data
    assert images_ref is doc.paste_images
    assert records_ref is doc.paste_records
    assert doc.pimg_data == []
    assert doc.real_pimg_data == []
    assert doc.paste_images == []
    assert doc.paste_records == []


def test_paste_document_append_paste_stores_aligned_state() -> None:
    doc = PasteDocument()
    record = SimpleNamespace(class_name="naval")

    state = doc.append_paste(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
        ["rgba", 0.1, 0.2, 0.3, 0.4],
        paste_record=record,
    )

    assert doc.total_pastes == 1
    assert state.real_row == ["naval", 10, 20, 30, 40]
    assert doc.pimg_data[0][0] == "naval"
    assert doc.real_pimg_data[0][0] == "naval"
    assert doc.paste_images[0][0] == "rgba"
    assert doc.paste_records[0].class_name == "naval"


def test_paste_document_replace_updates_sections_in_place() -> None:
    doc = PasteDocument()
    pimg_ref = doc.pimg_data
    real_ref = doc.real_pimg_data
    images_ref = doc.paste_images
    records_ref = doc.paste_records

    doc.replace(
        pimg_data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_pimg_data=[["naval", 10, 20, 30, 40]],
        paste_images=[["rgba", 0.1, 0.2, 0.3, 0.4]],
        paste_records=[SimpleNamespace(class_name="naval")],
    )

    assert pimg_ref is doc.pimg_data
    assert real_ref is doc.real_pimg_data
    assert images_ref is doc.paste_images
    assert records_ref is doc.paste_records
    assert doc.total_pastes == 1
    assert doc.pimg_data[0][:5] == ["naval", 1, 2, 3, 4]
    assert doc.real_pimg_data[0] == ["naval", 10, 20, 30, 40]
    assert doc.paste_records[0].class_name == "naval"


def test_paste_document_remove_and_rename_preserve_alignment() -> None:
    doc = PasteDocument()
    doc.append_paste(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
        ["rgba-1"],
        paste_record=SimpleNamespace(class_name="naval"),
    )
    doc.append_paste(
        ["merchant", 5, 6, 7, 8, 100, 100],
        ["merchant", 50, 60, 70, 80],
        ["rgba-2"],
        paste_record=SimpleNamespace(class_name="merchant"),
    )

    old_name = doc.rename_paste(0, "dock")

    assert old_name == "naval"
    assert doc.pimg_data[0][0] == "dock"
    assert doc.real_pimg_data[0][0] == "dock"
    assert doc.paste_records[0].class_name == "dock"

    removed = doc.remove_paste(0)

    assert removed.real_row[0] == "dock"
    assert doc.total_pastes == 1
    assert doc.real_pimg_data[0][0] == "merchant"
    doc.validate_alignment()


def test_paste_document_validate_alignment_rejects_mismatched_lengths() -> None:
    doc = PasteDocument(
        pimg_data=[["naval", 1, 2, 3, 4, 100, 100]],
        real_pimg_data=[],
        paste_images=[],
        paste_records=[],
    )

    with pytest.raises(ValueError, match="len\\(pimg_data\\)=1 != len\\(real_pimg_data\\)=0"):
        doc.validate_alignment()


def test_paste_document_append_paste_clones_outer_payload_without_deepcopying_inner_object() -> None:
    doc = PasteDocument()
    marker = _NoDeepcopy()
    payload = [marker, 0.1, 0.2, 0.3, 0.4]

    state = doc.append_paste(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
        payload,
    )

    assert state.paste_image is not payload
    assert doc.paste_images[0] is not payload
    assert state.paste_image[0] is marker
    assert doc.paste_images[0][0] is marker

    payload[1] = 9.9
    assert state.paste_image[1] == 0.1
    assert doc.paste_images[0][1] == 0.1
