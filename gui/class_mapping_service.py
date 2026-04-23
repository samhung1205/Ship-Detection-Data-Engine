"""
Load / resolve class catalog for the GUI (thin wrapper over sdde).
"""
from __future__ import annotations

from pathlib import Path

from sdde.class_catalog import ClassCatalog, default_ship_catalog
from sdde.classes_yaml import load_classes_yaml_path


def default_classes_yaml_path(
    project_root: Path | None = None,
    *,
    classes_yaml: str | Path = "classes.yaml",
) -> Path:
    root = project_root or Path(__file__).resolve().parent.parent
    path = Path(classes_yaml)
    if path.is_absolute():
        return path
    return root / path


def load_class_catalog(
    project_root: Path | None = None,
    *,
    classes_yaml: str | Path = "classes.yaml",
    classes_yaml_path: str | Path | None = None,
) -> ClassCatalog:
    path = Path(classes_yaml_path) if classes_yaml_path is not None else default_classes_yaml_path(
        project_root,
        classes_yaml=classes_yaml,
    )
    try:
        if path.exists():
            return load_classes_yaml_path(path)
    except (OSError, ValueError, KeyError):
        pass
    return default_ship_catalog()
