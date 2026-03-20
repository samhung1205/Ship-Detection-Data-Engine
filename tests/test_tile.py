"""Tests for tile grid computation, coord conversion, and annotations_in_tile."""

import pytest

from sdde.tile import (
    TileConfig,
    TileRect,
    annotations_in_tile,
    compute_tile_grid,
    global_to_tile,
    tile_to_global,
)


def test_single_tile_exactly_fits() -> None:
    cfg = TileConfig(tile_size=100, tile_stride=100)
    tiles = compute_tile_grid(100, 100, cfg)
    assert len(tiles) == 1
    t = tiles[0]
    assert (t.x, t.y, t.w, t.h) == (0, 0, 100, 100)


def test_grid_2x2() -> None:
    cfg = TileConfig(tile_size=100, tile_stride=100)
    tiles = compute_tile_grid(200, 200, cfg)
    assert len(tiles) == 4
    positions = [(t.x, t.y) for t in tiles]
    assert (0, 0) in positions
    assert (100, 0) in positions
    assert (0, 100) in positions
    assert (100, 100) in positions


def test_overlap_grid() -> None:
    cfg = TileConfig(tile_size=640, tile_stride=480)
    assert cfg.overlap == 160
    tiles = compute_tile_grid(960, 960, cfg)
    assert len(tiles) == 4


def test_last_tile_clamped() -> None:
    cfg = TileConfig(tile_size=100, tile_stride=80)
    tiles = compute_tile_grid(150, 100, cfg)
    last = [t for t in tiles if t.col == max(t2.col for t2 in tiles)][0]
    assert last.x + last.w <= 150


def test_zero_stride_returns_empty() -> None:
    cfg = TileConfig(tile_size=100, tile_stride=0)
    assert compute_tile_grid(100, 100, cfg) == []


def test_global_to_tile_and_back() -> None:
    t = TileRect(index=0, col=0, row=0, x=100, y=200, w=640, h=640)
    lx, ly = global_to_tile(300, 400, t)
    assert (lx, ly) == (200, 200)
    gx, gy = tile_to_global(lx, ly, t)
    assert (gx, gy) == (300, 400)


def test_annotations_in_tile_fully_inside() -> None:
    t = TileRect(index=0, col=0, row=0, x=0, y=0, w=100, h=100)
    real_data = [["ship", 10, 10, 50, 50]]
    result = annotations_in_tile(t, real_data)
    assert result == [0]


def test_annotations_in_tile_outside() -> None:
    t = TileRect(index=0, col=0, row=0, x=0, y=0, w=100, h=100)
    real_data = [["ship", 200, 200, 300, 300]]
    result = annotations_in_tile(t, real_data)
    assert result == []


def test_annotations_in_tile_partial() -> None:
    t = TileRect(index=0, col=0, row=0, x=0, y=0, w=100, h=100)
    real_data = [["ship", 80, 0, 200, 100]]
    inside = annotations_in_tile(t, real_data, min_visible_fraction=0.1)
    assert 0 in inside
    outside = annotations_in_tile(t, real_data, min_visible_fraction=0.5)
    assert 0 not in outside
