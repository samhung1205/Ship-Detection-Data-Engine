"""
Tests for the GT annotation preview controller.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.document import AnnotationDocument  # noqa: E402
from gui.annotation_preview_controller import AnnotationPreviewController  # noqa: E402


class _FakePixmap:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height

    def copy(self) -> "_FakePixmap":
        return _FakePixmap(self._width, self._height)


class _FakeImageCanvas:
    def __init__(self) -> None:
        self.painted = None
        self.sync_count = 0

    def paint_label_only(self, pixmap) -> None:
        self.painted = pixmap

    def sync_label_from_canvas(self) -> None:
        self.sync_count += 1


def test_preview_controller_draws_scaled_highlight() -> None:
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 10, 20, 30, 50, 100, 100],
        ["naval", 10, 20, 30, 50],
    )
    canvas = _FakePixmap(200, 100)
    image_canvas = _FakeImageCanvas()
    updates: list[str] = []
    controller = AnnotationPreviewController(
        document=doc,
        get_canvas=lambda: canvas,
        image_canvas=image_canvas,
        on_canvas_updated=lambda: updates.append("updated"),
    )

    calls: list[tuple[object, int, int, int, int, object, object]] = []
    import gui.annotation_preview_controller as preview_module

    orig_draw = preview_module.draw_selection_overlay
    orig_color = preview_module.QColor
    try:
        preview_module.QColor = lambda *rgba: rgba  # type: ignore[assignment]
        preview_module.draw_selection_overlay = lambda canvas_obj, *, x1, y1, x2, y2, fill_color, outline_color: calls.append(  # type: ignore[assignment]
            (canvas_obj, x1, y1, x2, y2, fill_color, outline_color)
        )
        controller.preview_row(0)
    finally:
        preview_module.draw_selection_overlay = orig_draw  # type: ignore[assignment]
        preview_module.QColor = orig_color  # type: ignore[assignment]

    assert image_canvas.painted is not None
    assert calls == [
        (
            image_canvas.painted,
            20,
            20,
            60,
            50,
            (30, 144, 255, 120),
            (30, 144, 255),
        )
    ]
    assert updates == ["updated"]


def test_preview_controller_clears_preview_for_invalid_row() -> None:
    image_canvas = _FakeImageCanvas()
    controller = AnnotationPreviewController(
        document=AnnotationDocument(),
        get_canvas=lambda: _FakePixmap(200, 100),
        image_canvas=image_canvas,
        on_canvas_updated=None,
    )

    controller.preview_row(0)

    assert image_canvas.painted is None
    assert image_canvas.sync_count == 1


def test_preview_controller_can_clear_preview_explicitly() -> None:
    image_canvas = _FakeImageCanvas()
    updates: list[str] = []
    controller = AnnotationPreviewController(
        document=AnnotationDocument(),
        get_canvas=lambda: None,
        image_canvas=image_canvas,
        on_canvas_updated=lambda: updates.append("updated"),
    )

    controller.clear_preview()

    assert image_canvas.sync_count == 1
    assert updates == ["updated"]
