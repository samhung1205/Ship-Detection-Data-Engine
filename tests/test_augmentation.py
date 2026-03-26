"""Tests for augmentation metadata (PasteRecord, JSON/CSV export)."""

import json

from sdde.augmentation import (
    PasteRecord,
    bbox_from_legacy_paste_row,
    export_paste_records_csv,
    export_paste_records_json,
)


def _sample() -> PasteRecord:
    return PasteRecord(
        paste_id="abc123",
        image_path="img.jpg",
        asset_path="ship.png",
        class_name="ship",
        scale=1.5,
        rotation_deg=45.0,
        h_flip=True,
        brightness=110,
        contrast=90,
        bbox_x1=10,
        bbox_y1=20,
        bbox_x2=100,
        bbox_y2=200,
        timestamp="2026-01-01T00:00:00+00:00",
    )


def test_paste_record_defaults() -> None:
    r = PasteRecord()
    assert r.scale == 1.0
    assert r.h_flip is False
    assert r.paste_id  # non-empty


def test_export_json() -> None:
    recs = [_sample()]
    text = export_paste_records_json(recs)
    data = json.loads(text)
    assert len(data) == 1
    assert data[0]["class_name"] == "ship"
    assert data[0]["rotation_deg"] == 45.0
    assert data[0]["h_flip"] is True


def test_export_csv() -> None:
    recs = [_sample(), _sample()]
    text = export_paste_records_csv(recs)
    lines = text.strip().splitlines()
    assert len(lines) == 3  # header + 2 rows
    assert "asset_path" in lines[0]
    assert "ship.png" in lines[1]


def test_export_empty() -> None:
    assert export_paste_records_json([]) == "[]"
    csv_text = export_paste_records_csv([])
    lines = csv_text.strip().splitlines()
    assert len(lines) == 1  # header only


def test_bbox_from_legacy_paste_row_accepts_class_prefixed_row() -> None:
    assert bbox_from_legacy_paste_row(["naval", 10, 20, 100, 200]) == (10, 20, 100, 200)


def test_bbox_from_legacy_paste_row_accepts_plain_bbox_row() -> None:
    assert bbox_from_legacy_paste_row([10, 20, 100, 200]) == (10, 20, 100, 200)


def test_bbox_from_legacy_paste_row_rejects_non_numeric_bbox_values() -> None:
    try:
        bbox_from_legacy_paste_row(["naval", "x", 20, 100, 200])
    except ValueError as exc:
        assert "non-integer" in str(exc)
    else:
        raise AssertionError("Expected ValueError for malformed paste bbox row.")
