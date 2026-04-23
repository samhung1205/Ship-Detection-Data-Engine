"""Tests for supported-image folder browsing helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.image_browser import find_image_index, list_supported_images  # noqa: E402


def test_list_supported_images_filters_and_sorts_supported_files(tmp_path: Path) -> None:
    (tmp_path / "b.png").write_text("x", encoding="utf-8")
    (tmp_path / "A.TIFF").write_text("x", encoding="utf-8")
    (tmp_path / "note.txt").write_text("x", encoding="utf-8")
    (tmp_path / "c.JPG").write_text("x", encoding="utf-8")

    result = list_supported_images(tmp_path)

    assert result == [
        str(tmp_path / "A.TIFF"),
        str(tmp_path / "b.png"),
        str(tmp_path / "c.JPG"),
    ]


def test_find_image_index_returns_minus_one_for_missing_path() -> None:
    paths = ["/tmp/a.jpg", "/tmp/b.tif"]

    assert find_image_index(paths, "/tmp/b.tif") == 1
    assert find_image_index(paths, "/tmp/missing.png") == -1
