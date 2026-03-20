"""
Load / resolve class catalog for the GUI (thin wrapper over sdde).
"""
from __future__ import annotations

from pathlib import Path

from sdde.class_catalog import ClassCatalog, default_ship_catalog
from sdde.classes_yaml import load_classes_yaml_path


def default_classes_yaml_path(project_root: Path | None = None) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent
    return root / "classes.yaml"


def load_class_catalog(project_root: Path | None = None) -> ClassCatalog:
    path = default_classes_yaml_path(project_root)
    try:
        if path.exists():
            return load_classes_yaml_path(path)
    except (OSError, ValueError, KeyError):
        pass
    return default_ship_catalog()
