"""
Project configuration (PRD §10.1 / feature list §4.6).

Persists project-level settings to a YAML file so the user doesn't have to
reconfigure paths, tile sizes, class mapping, etc. on every launch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml


@dataclass
class ProjectConfig:
    project_name: str = "ship_detection_project"
    project_root: str = "."
    image_root: str = "dataset"
    label_root: str = "dataset"
    classes_yaml: str = "classes.yaml"
    default_export_format: str = "yolo_hbb"
    autosave_seconds: int = 60
    recent_images: list[str] = field(default_factory=list)
    tile_size: int = 640
    tile_stride: int = 480

    MAX_RECENT = 20

    def add_recent_image(self, path: str) -> None:
        if path in self.recent_images:
            self.recent_images.remove(path)
        self.recent_images.insert(0, path)
        if len(self.recent_images) > self.MAX_RECENT:
            self.recent_images = self.recent_images[: self.MAX_RECENT]


def _to_dict(cfg: ProjectConfig) -> dict[str, Any]:
    return {
        "project_name": cfg.project_name,
        "project_root": cfg.project_root,
        "image_root": cfg.image_root,
        "label_root": cfg.label_root,
        "classes_yaml": cfg.classes_yaml,
        "default_export_format": cfg.default_export_format,
        "autosave_seconds": cfg.autosave_seconds,
        "recent_images": cfg.recent_images,
        "tile_size": cfg.tile_size,
        "tile_stride": cfg.tile_stride,
    }


def save_project_config(cfg: ProjectConfig, path: str | Path) -> None:
    Path(path).write_text(
        yaml.safe_dump(_to_dict(cfg), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def load_project_config(path: str | Path) -> ProjectConfig:
    p = Path(path)
    raw: Any = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("project_config.yaml root must be a mapping")
    return ProjectConfig(
        project_name=str(raw.get("project_name", "ship_detection_project")),
        project_root=str(raw.get("project_root", ".")),
        image_root=str(raw.get("image_root", "dataset")),
        label_root=str(raw.get("label_root", "dataset")),
        classes_yaml=str(raw.get("classes_yaml", "classes.yaml")),
        default_export_format=str(raw.get("default_export_format", "yolo_hbb")),
        autosave_seconds=int(raw.get("autosave_seconds", 60)),
        recent_images=list(raw.get("recent_images") or []),
        tile_size=int(raw.get("tile_size", 640)),
        tile_stride=int(raw.get("tile_stride", 480)),
    )
