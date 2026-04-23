"""
Helpers for browsing supported image files in a folder.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable


SUPPORTED_IMAGE_SUFFIXES = (
    ".jpg",
    ".jpeg",
    ".png",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
)


def is_supported_image_path(path: str | Path) -> bool:
    return Path(path).suffix.lower() in SUPPORTED_IMAGE_SUFFIXES


def list_supported_images(folder: str | Path) -> list[str]:
    root = Path(folder)
    if not root.is_dir():
        return []
    images = [
        str(path)
        for path in root.iterdir()
        if path.is_file() and is_supported_image_path(path)
    ]
    return sorted(images, key=lambda value: Path(value).name.lower())


def find_image_index(image_paths: Iterable[str], current_path: str | Path) -> int:
    current = str(Path(current_path))
    for idx, path in enumerate(image_paths):
        if str(Path(path)) == current:
            return idx
    return -1
