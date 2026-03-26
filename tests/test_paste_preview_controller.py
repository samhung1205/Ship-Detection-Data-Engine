"""
Tests for the paste preview controller.
"""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.paste_preview_controller import PastePreviewController  # noqa: E402
from sdde.paste_document import PasteDocument  # noqa: E402


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


class _FakePainter:
    last_instance = None

    def __init__(self) -> None:
        self.begin_target = None
        self.fill_calls = []
        self.ended = False
        type(self).last_instance = self

    def begin(self, target) -> bool:
        self.begin_target = target
        return True

    def fillRect(self, x: int, y: int, width: int, height: int, color) -> None:  # noqa: N802
        self.fill_calls.append((x, y, width, height, color))

    def end(self) -> None:
        self.ended = True


def test_paste_preview_controller_draws_scaled_highlight() -> None:
    document = PasteDocument(
        pimg_data=[["naval", 10, 20, 30, 50, 100, 100]],
        real_pimg_data=[["naval", 100, 200, 300, 500]],
        paste_images=[["rgba"]],
    )
    canvas = _FakePixmap(200, 100)
    image_canvas = _FakeImageCanvas()
    updates: list[str] = []
    controller = PastePreviewController(
        document=document,
        get_canvas=lambda: canvas,
        image_canvas=image_canvas,
        on_canvas_updated=lambda: updates.append("updated"),
    )

    with patch("gui.paste_preview_controller.QPainter", _FakePainter):
        with patch("gui.paste_preview_controller.QColor", lambda *rgba: rgba):
            controller.preview_row(0)

    painter = _FakePainter.last_instance
    assert painter is not None
    assert painter.begin_target is image_canvas.painted
    assert painter.fill_calls == [(20, 20, 41, 31, (30, 144, 255, 120))]
    assert painter.ended is True
    assert updates == ["updated"]


def test_paste_preview_controller_clears_preview_for_invalid_row() -> None:
    image_canvas = _FakeImageCanvas()
    controller = PastePreviewController(
        document=PasteDocument(),
        get_canvas=lambda: _FakePixmap(200, 100),
        image_canvas=image_canvas,
        on_canvas_updated=None,
    )

    controller.preview_row(0)

    assert image_canvas.painted is None
    assert image_canvas.sync_count == 1


def test_paste_preview_controller_can_clear_preview_explicitly() -> None:
    image_canvas = _FakeImageCanvas()
    updates: list[str] = []
    controller = PastePreviewController(
        document=PasteDocument(),
        get_canvas=lambda: None,
        image_canvas=image_canvas,
        on_canvas_updated=lambda: updates.append("updated"),
    )

    controller.clear_preview()

    assert image_canvas.sync_count == 1
    assert updates == ["updated"]
