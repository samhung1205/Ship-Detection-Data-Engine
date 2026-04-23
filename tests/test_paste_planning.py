"""Tests for smart-paste planning helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.paste_planning import (  # noqa: E402
    normalize_rect,
    rect_contains,
    scale_hint_for_size_tag,
    size_tag_for_scale_factor,
)


def test_normalize_rect_orders_coordinates() -> None:
    assert normalize_rect(40, 50, 10, 20) == (10, 20, 40, 50)


def test_rect_contains_checks_inner_bbox() -> None:
    assert rect_contains((10, 10, 80, 80), (20, 20, 60, 60)) is True
    assert rect_contains((10, 10, 80, 80), (5, 20, 60, 60)) is False


def test_scale_hint_for_medium_returns_bounded_range() -> None:
    hint = scale_hint_for_size_tag(40, 20, "medium", export_scale=1.0)

    assert hint.reachable is True
    assert hint.min_factor is not None
    assert hint.max_factor is not None
    assert hint.min_factor < hint.max_factor


def test_scale_hint_for_large_can_be_unreachable() -> None:
    hint = scale_hint_for_size_tag(8, 8, "large", export_scale=0.1)

    assert hint.reachable is False
    assert hint.min_factor == 10.0
    assert hint.max_factor is None


def test_size_tag_for_scale_factor_tracks_current_scale() -> None:
    assert size_tag_for_scale_factor(20, 20, 1.0) == "small"
    assert size_tag_for_scale_factor(20, 20, 2.0) == "medium"
