"""Tests for data.yaml compatibility helpers."""

from pathlib import Path

from sdde.config import class_mapping_from_data_yaml


ROOT = Path(__file__).resolve().parents[1]


def test_load_repo_data_yaml() -> None:
    mapping = class_mapping_from_data_yaml(ROOT / "data.yaml")
    assert mapping.names == ["naval", "merchant", "dock", "other_vessel"]
