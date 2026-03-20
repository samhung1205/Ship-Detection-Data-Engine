"""
Tile / sliding-window utilities (PRD §15).

Tiles are a *view* over the full image — all annotations remain in global
(origin-pixel) coordinates. The tile grid is computed from image size,
tile_size and tile_stride; overlap = tile_size - tile_stride.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass
class TileRect:
    """One tile in origin-pixel space (top-left inclusive, bottom-right exclusive)."""
    index: int
    col: int
    row: int
    x: int
    y: int
    w: int
    h: int

    @property
    def x2(self) -> int:
        return self.x + self.w

    @property
    def y2(self) -> int:
        return self.y + self.h


@dataclass
class TileConfig:
    tile_size: int = 640
    tile_stride: int = 480

    @property
    def overlap(self) -> int:
        return max(0, self.tile_size - self.tile_stride)


def compute_tile_grid(
    image_w: int,
    image_h: int,
    cfg: TileConfig,
) -> list[TileRect]:
    """
    Generate a row-major list of tile rectangles that cover the full image.

    The last column / row is clamped so it doesn't exceed image bounds.
    """
    if cfg.tile_size <= 0 or cfg.tile_stride <= 0:
        return []

    tiles: list[TileRect] = []
    idx = 0
    r = 0
    y = 0
    while y < image_h:
        c = 0
        x = 0
        th = min(cfg.tile_size, image_h - y)
        while x < image_w:
            tw = min(cfg.tile_size, image_w - x)
            tiles.append(TileRect(index=idx, col=c, row=r, x=x, y=y, w=tw, h=th))
            idx += 1
            c += 1
            x += cfg.tile_stride
            if x >= image_w:
                break
        r += 1
        y += cfg.tile_stride
        if y >= image_h:
            break
    return tiles


def annotations_in_tile(
    tile: TileRect,
    real_data: Sequence[list],
    *,
    min_visible_fraction: float = 0.25,
) -> list[int]:
    """
    Return indices of real_data rows whose bbox overlaps the tile by at least
    *min_visible_fraction* of the bbox area.  Useful for filtering the
    annotation list when viewing a tile.

    Each real_data row is [class_name, x1, y1, x2, y2].
    """
    result: list[int] = []
    tx1, ty1, tx2, ty2 = tile.x, tile.y, tile.x2, tile.y2
    for i, row in enumerate(real_data):
        bx1, by1, bx2, by2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
        ix1 = max(bx1, tx1)
        iy1 = max(by1, ty1)
        ix2 = min(bx2, tx2)
        iy2 = min(by2, ty2)
        inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        bbox_area = max(1e-9, (bx2 - bx1) * (by2 - by1))
        if inter / bbox_area >= min_visible_fraction:
            result.append(i)
    return result


def global_to_tile(
    gx: float, gy: float, tile: TileRect
) -> tuple[float, float]:
    """Convert origin-pixel global coords to tile-local coords."""
    return gx - tile.x, gy - tile.y


def tile_to_global(
    tx: float, ty: float, tile: TileRect
) -> tuple[float, float]:
    """Convert tile-local coords back to origin-pixel global coords."""
    return tx + tile.x, ty + tile.y
