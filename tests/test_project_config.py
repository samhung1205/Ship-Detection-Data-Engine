"""Tests for ProjectConfig load / save / recent_images."""

import pytest
from pathlib import Path

from sdde.project_config import ProjectConfig, load_project_config, save_project_config


def test_defaults() -> None:
    cfg = ProjectConfig()
    assert cfg.autosave_seconds == 60
    assert cfg.tile_size == 640
    assert cfg.tile_stride == 480
    assert cfg.recent_images == []


def test_add_recent_image() -> None:
    cfg = ProjectConfig()
    cfg.add_recent_image("a.jpg")
    cfg.add_recent_image("b.jpg")
    assert cfg.recent_images == ["b.jpg", "a.jpg"]
    cfg.add_recent_image("a.jpg")
    assert cfg.recent_images == ["a.jpg", "b.jpg"]


def test_add_recent_max() -> None:
    cfg = ProjectConfig()
    for i in range(25):
        cfg.add_recent_image(f"{i}.jpg")
    assert len(cfg.recent_images) == cfg.MAX_RECENT


def test_round_trip(tmp_path: Path) -> None:
    cfg = ProjectConfig(
        project_name="test_proj",
        image_root="/data/images",
        label_root="/data/labels",
        autosave_seconds=30,
        tile_size=320,
        tile_stride=160,
    )
    cfg.add_recent_image("foo.jpg")
    fp = tmp_path / "cfg.yaml"
    save_project_config(cfg, fp)
    loaded = load_project_config(fp)
    assert loaded.project_name == "test_proj"
    assert loaded.autosave_seconds == 30
    assert loaded.tile_size == 320
    assert loaded.tile_stride == 160
    assert loaded.recent_images == ["foo.jpg"]


def test_load_missing_file(tmp_path: Path) -> None:
    with pytest.raises(OSError):
        load_project_config(tmp_path / "nonexistent.yaml")
