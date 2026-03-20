"""
Autosave service (PRD §5.9 / §18.1).

Periodically saves current annotations to a JSON sidecar file next to the
original image, so work can be recovered after an unexpected crash.

File format: ``<image_stem>.autosave.json`` in a ``.autosave/`` directory
under the project root (or next to the image).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence


AUTOSAVE_DIR = ".autosave"
AUTOSAVE_SUFFIX = ".autosave.json"


def _autosave_path(image_path: str, autosave_root: str | None = None) -> Path:
    img = Path(image_path)
    root = Path(autosave_root) if autosave_root else img.parent
    d = root / AUTOSAVE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d / (img.stem + AUTOSAVE_SUFFIX)


def write_autosave(
    image_path: str,
    real_data: Sequence[Sequence[Any]],
    box_attributes: Sequence[Mapping[str, str]],
    object_list: Sequence[str],
    *,
    autosave_root: str | None = None,
) -> Path:
    """Write current annotation state to the autosave sidecar file."""
    payload = {
        "image_path": image_path,
        "object_list": list(object_list),
        "real_data": [list(r) for r in real_data],
        "box_attributes": [dict(a) for a in box_attributes],
    }
    fp = _autosave_path(image_path, autosave_root)
    fp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return fp


def read_autosave(
    image_path: str,
    *,
    autosave_root: str | None = None,
) -> dict[str, Any] | None:
    """
    Read autosave data if it exists for the given image.

    Returns dict with keys: image_path, object_list, real_data, box_attributes.
    Returns None if no autosave file is found.
    """
    fp = _autosave_path(image_path, autosave_root)
    if not fp.exists():
        return None
    try:
        return json.loads(fp.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def remove_autosave(
    image_path: str,
    *,
    autosave_root: str | None = None,
) -> None:
    """Delete autosave file for the given image (called after a successful manual save)."""
    fp = _autosave_path(image_path, autosave_root)
    if fp.exists():
        fp.unlink(missing_ok=True)


def has_autosave(
    image_path: str,
    *,
    autosave_root: str | None = None,
) -> bool:
    return _autosave_path(image_path, autosave_root).exists()
