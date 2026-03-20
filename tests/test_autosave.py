"""Tests for autosave write / read / remove / has."""

from pathlib import Path

from sdde.autosave import has_autosave, read_autosave, remove_autosave, write_autosave


def test_write_and_read(tmp_path: Path) -> None:
    img = str(tmp_path / "image.jpg")
    real_data = [["ship", 10, 20, 30, 40]]
    box_attrs = [{"size_tag": "small", "crowded": "false"}]
    obj_list = ["ship"]

    fp = write_autosave(img, real_data, box_attrs, obj_list, autosave_root=str(tmp_path))
    assert fp.exists()
    assert has_autosave(img, autosave_root=str(tmp_path))

    data = read_autosave(img, autosave_root=str(tmp_path))
    assert data is not None
    assert data["image_path"] == img
    assert data["real_data"] == [["ship", 10, 20, 30, 40]]
    assert data["box_attributes"][0]["size_tag"] == "small"


def test_remove(tmp_path: Path) -> None:
    img = str(tmp_path / "image.jpg")
    write_autosave(img, [], [], [], autosave_root=str(tmp_path))
    assert has_autosave(img, autosave_root=str(tmp_path))
    remove_autosave(img, autosave_root=str(tmp_path))
    assert not has_autosave(img, autosave_root=str(tmp_path))


def test_read_nonexistent(tmp_path: Path) -> None:
    img = str(tmp_path / "nope.jpg")
    assert read_autosave(img, autosave_root=str(tmp_path)) is None
    assert not has_autosave(img, autosave_root=str(tmp_path))


def test_overwrite(tmp_path: Path) -> None:
    img = str(tmp_path / "img.jpg")
    write_autosave(img, [["a", 1, 2, 3, 4]], [], ["a"], autosave_root=str(tmp_path))
    write_autosave(img, [["b", 5, 6, 7, 8]], [], ["b"], autosave_root=str(tmp_path))
    data = read_autosave(img, autosave_root=str(tmp_path))
    assert data is not None
    assert data["real_data"][0][0] == "b"
