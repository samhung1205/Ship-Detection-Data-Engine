"""
Load / save PRD-style classes.yaml (project_name + classes list).
Requires PyYAML (see requirements.txt).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from .class_catalog import ClassCatalog, ClassInfo


def load_classes_yaml(text: str) -> ClassCatalog:
    raw: Any = yaml.safe_load(text)
    if not isinstance(raw, Mapping):
        raise ValueError("classes.yaml root must be a mapping")
    project_name = str(raw.get("project_name") or "ship_detection_project")
    rows = raw.get("classes")
    if not isinstance(rows, list):
        raise ValueError("classes.yaml must contain a 'classes' list")
    classes: list[ClassInfo] = []
    for item in rows:
        if not isinstance(item, Mapping):
            raise ValueError("Each class entry must be a mapping")
        classes.append(
            ClassInfo(
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                super_category=str(item["super_category"]),
            )
        )
    cat = ClassCatalog.from_list(project_name, classes)
    cat.validate()
    return cat


def load_classes_yaml_path(path: str | Path) -> ClassCatalog:
    p = Path(path)
    return load_classes_yaml(p.read_text(encoding="utf-8"))


def save_classes_yaml(cat: ClassCatalog) -> str:
    data = {
        "project_name": cat.project_name,
        "classes": [
            {
                "class_id": c.class_id,
                "class_name": c.class_name,
                "super_category": c.super_category,
            }
            for c in sorted(cat.classes, key=lambda c: c.class_id)
        ],
    }
    return yaml.safe_dump(data, sort_keys=False, allow_unicode=True)


def save_classes_yaml_path(cat: ClassCatalog, path: str | Path) -> None:
    Path(path).write_text(save_classes_yaml(cat), encoding="utf-8")
