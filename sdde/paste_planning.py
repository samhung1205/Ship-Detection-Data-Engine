"""
Planning helpers for smart paste placement and size hints.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from sdde.attributes import LARGE_AREA_MIN, SMALL_AREA_MAX, compute_size_tag


MIN_SCALE_FACTOR = 0.1
MAX_SCALE_FACTOR = 10.0


@dataclass(frozen=True)
class PasteScaleHint:
    min_factor: float | None
    max_factor: float | None
    reachable: bool = True


def normalize_rect(
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> tuple[int, int, int, int]:
    return (
        min(int(x1), int(x2)),
        min(int(y1), int(y2)),
        max(int(x1), int(x2)),
        max(int(y1), int(y2)),
    )


def rect_contains(
    outer: tuple[int, int, int, int] | None,
    inner: tuple[int, int, int, int] | list[int] | None,
) -> bool:
    if outer is None or inner is None or len(inner) < 4:
        return False
    ox1, oy1, ox2, oy2 = outer
    ix1, iy1, ix2, iy2 = [int(v) for v in inner[:4]]
    return ox1 <= ix1 and oy1 <= iy1 and ox2 >= ix2 and oy2 >= iy2


def scale_hint_for_size_tag(
    source_width: int,
    source_height: int,
    target_size_tag: str,
    *,
    export_scale: float = 1.0,
    min_scale_factor: float = MIN_SCALE_FACTOR,
    max_scale_factor: float = MAX_SCALE_FACTOR,
) -> PasteScaleHint:
    base_area = (
        float(max(0, source_width))
        * float(max(0, source_height))
        * float(max(export_scale, 0.0) ** 2)
    )
    if base_area <= 0:
        return PasteScaleHint(None, None, reachable=False)

    if target_size_tag == "small":
        upper = math.sqrt(SMALL_AREA_MAX / base_area)
        if upper < min_scale_factor:
            return PasteScaleHint(None, min_scale_factor, reachable=False)
        return PasteScaleHint(None, min(max_scale_factor, upper), reachable=True)

    if target_size_tag == "large":
        lower = math.sqrt(LARGE_AREA_MIN / base_area)
        if lower > max_scale_factor:
            return PasteScaleHint(max_scale_factor, None, reachable=False)
        return PasteScaleHint(max(min_scale_factor, lower), None, reachable=True)

    lower = math.sqrt(SMALL_AREA_MAX / base_area)
    upper = math.sqrt(LARGE_AREA_MIN / base_area)
    clipped_lower = max(min_scale_factor, lower)
    clipped_upper = min(max_scale_factor, upper)
    return PasteScaleHint(
        clipped_lower,
        clipped_upper,
        reachable=clipped_lower <= clipped_upper,
    )


def size_tag_for_scale_factor(
    source_width: int,
    source_height: int,
    scale_factor: float,
    *,
    export_scale: float = 1.0,
) -> str:
    width = float(max(0, source_width)) * float(max(scale_factor, 0.0)) * float(max(export_scale, 0.0))
    height = float(max(0, source_height)) * float(max(scale_factor, 0.0)) * float(max(export_scale, 0.0))
    return compute_size_tag(0.0, 0.0, width, height)
