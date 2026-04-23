"""Tests for low-level canvas drawing helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import gui.canvas_utils as canvas_utils  # noqa: E402
from sdde.error_analysis import ErrorCase  # noqa: E402


class _FakeCanvas:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _FakePixmap:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height
        self.copy_args = None

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def copy(self, x: int, y: int, width: int, height: int):
        self.copy_args = (x, y, width, height)
        return _FakePixmap(width, height)

    def scaled(self, width: int, height: int, *_args):
        return _FakePixmap(width, height)


class _FakePainter:
    instances: list["_FakePainter"] = []
    class RenderHint:
        SmoothPixmapTransform = object()

    def __init__(self) -> None:
        self.ops: list[tuple[str, tuple]] = []
        _FakePainter.instances.append(self)

    def begin(self, _canvas) -> None:
        self.ops.append(("begin", ()))

    def setPen(self, _pen) -> None:  # noqa: N802
        self.ops.append(("setPen", ()))

    def setRenderHint(self, *_args) -> None:  # noqa: N802
        self.ops.append(("setRenderHint", ()))

    def setFont(self, _font) -> None:  # noqa: N802
        self.ops.append(("setFont", ()))

    def fillRect(self, *args) -> None:  # noqa: N802
        self.ops.append(("fillRect", args))

    def drawRect(self, *args) -> None:  # noqa: N802
        self.ops.append(("drawRect", args))

    def drawText(self, *args) -> None:  # noqa: N802
        self.ops.append(("drawText", args))

    def drawImage(self, *args) -> None:  # noqa: N802
        self.ops.append(("drawImage", args))

    def drawLine(self, *args) -> None:  # noqa: N802
        self.ops.append(("drawLine", args))

    def end(self) -> None:
        self.ops.append(("end", ()))


def test_draw_boundary_boxes_on_canvas_scales_and_labels() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_boundary_boxes_on_canvas(
            _FakeCanvas(200, 100),
            [["ship", 10, 20, 30, 50]],
            origin_width=100,
            origin_height=100,
            labels=["#1"],
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    painter = _FakePainter.instances[0]
    assert ("drawRect", (20, 20, 40, 30)) in painter.ops
    assert ("drawText", (22, 18, "#1")) in painter.ops


def test_draw_boundary_boxes_on_canvas_skips_empty_input() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_boundary_boxes_on_canvas(
            _FakeCanvas(200, 100),
            [],
            origin_width=100,
            origin_height=100,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    assert _FakePainter.instances == []


def test_draw_tile_grid_overview_highlights_current_tile() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_tile_grid_overview(
            _FakeCanvas(200, 100),
            [(0, 0, 100, 100), (100, 0, 100, 100)],
            current_index=1,
            origin_width=200,
            origin_height=100,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    painter = _FakePainter.instances[0]
    draw_rect_ops = [op for op in painter.ops if op[0] == "drawRect"]
    assert ("drawRect", (0, 0, 100, 100)) in draw_rect_ops
    assert ("drawRect", (100, 0, 100, 100)) in draw_rect_ops
    assert ("drawText", (104, 12, "Tile #2")) in painter.ops


def test_draw_paste_images_on_canvas_can_use_export_geometry_payload() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_paste_images_on_canvas(
            _FakeCanvas(100, 50),
            [["display", 0.1, 0.2, 0.3, 0.4, "export", 0.5, 0.1, 0.2, 0.6]],
            prefer_export_geometry=True,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    painter = _FakePainter.instances[0]
    draw_image = next(op for op in painter.ops if op[0] == "drawImage")
    rect = draw_image[1][0]
    image = draw_image[1][1]
    assert rect.x() == 50
    assert rect.y() == 5
    assert rect.width() == 20
    assert rect.height() == 30
    assert image == "export"


def test_draw_paste_zone_overlay_scales_and_labels() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_paste_zone_overlay(
            _FakeCanvas(200, 100),
            (10, 20, 60, 80),
            origin_width=100,
            origin_height=100,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    painter = _FakePainter.instances[0]
    assert ("drawRect", (20, 20, 100, 60)) in painter.ops
    assert ("drawText", (23, 18, "Smart zone")) in painter.ops


def test_compute_magnifier_source_rect_clamps_to_edges() -> None:
    assert canvas_utils.compute_magnifier_source_rect(
        center_x=5,
        center_y=8,
        canvas_width=100,
        canvas_height=80,
        preview_size=160,
        zoom_factor=4.0,
    ) == (0, 0, 40, 40)

    assert canvas_utils.compute_magnifier_source_rect(
        center_x=95,
        center_y=78,
        canvas_width=100,
        canvas_height=80,
        preview_size=160,
        zoom_factor=4.0,
    ) == (60, 40, 40, 40)


def test_compute_magnifier_anchor_flips_near_bottom_right() -> None:
    anchor = canvas_utils.compute_magnifier_anchor(
        cursor_x=190,
        cursor_y=180,
        image_width=220,
        image_height=220,
        preview_size=80,
        offset=10,
    )
    assert anchor == (100, 90)


def test_build_magnifier_preview_crops_and_draws_crosshair() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        source = _FakePixmap(200, 120)
        preview = canvas_utils.build_magnifier_preview(
            source,
            center_x=100,
            center_y=60,
            preview_size=120,
            zoom_factor=3.0,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    assert preview is not None
    assert preview.width() == 120
    assert preview.height() == 120
    assert source.copy_args == (80, 40, 40, 40)
    painter = _FakePainter.instances[0]
    assert ("drawLine", (60, 0, 60, 120)) in painter.ops
    assert ("drawLine", (0, 60, 120, 60)) in painter.ops
    assert ("drawRect", (0, 0, 119, 119)) in painter.ops


def test_draw_error_cases_overlay_draws_pair_line_and_fp_fn_badges() -> None:
    original_painter = canvas_utils.QPainter
    _FakePainter.instances.clear()
    try:
        canvas_utils.QPainter = _FakePainter  # type: ignore[assignment]
        canvas_utils.draw_error_cases_overlay(
            _FakeCanvas(200, 100),
            [
                ErrorCase(error_type="TP", gt_index=0, pred_index=0, iou=0.75),
                ErrorCase(error_type="FP", pred_index=1),
                ErrorCase(error_type="FN", gt_index=1),
            ],
            gt_boxes=[
                ("ship", 0, 0, 100, 100),
                ("ship", 100, 0, 200, 100),
            ],
            predictions=[
                type("P", (), {"x1": 10, "y1": 10, "x2": 90, "y2": 90})(),
                type("P", (), {"x1": 150, "y1": 10, "x2": 190, "y2": 50})(),
            ],
            origin_width=200,
            origin_height=100,
        )
    finally:
        canvas_utils.QPainter = original_painter  # type: ignore[assignment]

    painter = _FakePainter.instances[0]
    assert any(op[0] == "drawLine" for op in painter.ops)
    assert ("drawText", (50, 50, "TP 0.75")) in painter.ops
    assert ("drawText", (152, 12, "FP")) in painter.ops
    assert ("drawText", (102, 12, "FN")) in painter.ops
