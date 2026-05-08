"""Tests for class mapping service behavior."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.class_mapping_service import load_class_catalog


def test_load_class_catalog_falls_back_only_when_file_is_missing(tmp_path: Path) -> None:
    catalog = load_class_catalog(classes_yaml_path=tmp_path / "missing.yaml")

    assert list(catalog.names_ordered()) == ["naval", "merchant", "dock", "other_vessel"]


def test_load_class_catalog_raises_for_invalid_yaml(tmp_path: Path) -> None:
    classes_path = tmp_path / "classes.yaml"
    classes_path.write_text("classes: not-a-list\n", encoding="utf-8")

    with pytest.raises(ValueError):
        load_class_catalog(classes_yaml_path=classes_path)
