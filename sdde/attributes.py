"""
Research attribute fields (PRD §9). Values stored as strings on each box annotation.
"""
from __future__ import annotations

from typing import Dict, List, Mapping

# PRD §9.5 COCO-style area thresholds (pixel space, axis-aligned box)
SMALL_AREA_MAX = 32 * 32  # exclusive upper bound for small
LARGE_AREA_MIN = 96 * 96  # exclusive lower bound for large → large if area > this

SIZE_TAG_CHOICES: List[str] = ["small", "medium", "large"]
CROWDED_CHOICES: List[str] = ["false", "true"]
DIFFICULTY_CHOICES: List[str] = ["normal", "hard", "uncertain"]
SCENE_CHOICES: List[str] = ["near_shore", "offshore", "unknown"]

# Keys used in GUI / export
KEY_SIZE = "size_tag"
KEY_CROWDED = "crowded"
KEY_DIFFICULTY = "difficulty_tag"
KEY_SCENE = "scene_tag"
KEY_HARD_SAMPLE = "hard_sample"
KEY_OCCLUDED = "occluded"
KEY_TRUNCATED = "truncated"
KEY_BLURRED = "blurred"
KEY_DIFFICULT_BACKGROUND = "difficult_background"
KEY_LOW_CONTRAST = "low_contrast"

_BOOLEAN_KEYS = {
    KEY_CROWDED,
    KEY_HARD_SAMPLE,
    KEY_OCCLUDED,
    KEY_TRUNCATED,
    KEY_BLURRED,
    KEY_DIFFICULT_BACKGROUND,
    KEY_LOW_CONTRAST,
}
_TRUE_VALUES = {"1", "true", "yes", "y", "on"}
_FALSE_VALUES = {"0", "false", "no", "n", "off"}


def _normalize_bool_flag(value: object) -> str:
    text = str(value).strip().lower()
    if text in _TRUE_VALUES:
        return "true"
    if text in _FALSE_VALUES or not text:
        return "false"
    return "false"


def bbox_area_px(x1: float, y1: float, x2: float, y2: float) -> float:
    w = abs(float(x2) - float(x1))
    h = abs(float(y2) - float(y1))
    return w * h


def compute_size_tag(x1: float, y1: float, x2: float, y2: float) -> str:
    """PRD §9.5 using origin-pixel axis-aligned area."""
    a = bbox_area_px(x1, y1, x2, y2)
    if a < SMALL_AREA_MAX:
        return "small"
    if a <= LARGE_AREA_MIN:
        return "medium"
    return "large"


def default_attributes_dict() -> Dict[str, str]:
    return {
        KEY_SIZE: "medium",
        KEY_CROWDED: "false",
        KEY_DIFFICULTY: "normal",
        KEY_SCENE: "unknown",
        KEY_HARD_SAMPLE: "false",
        KEY_OCCLUDED: "false",
        KEY_TRUNCATED: "false",
        KEY_BLURRED: "false",
        KEY_DIFFICULT_BACKGROUND: "false",
        KEY_LOW_CONTRAST: "false",
    }


def normalize_attributes(m: Mapping[str, str]) -> Dict[str, str]:
    """Fill missing keys with defaults."""
    out = default_attributes_dict()
    for k in out:
        if k not in m:
            continue
        value = m[k]
        if k in _BOOLEAN_KEYS:
            out[k] = _normalize_bool_flag(value)
            continue
        if str(value).strip():
            out[k] = str(value).strip()
    return out


def attributes_to_flat_dict(attrs: Mapping[str, str]) -> Dict[str, str]:
    return normalize_attributes(attrs)
