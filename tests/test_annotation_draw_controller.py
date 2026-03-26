"""
Tests for the GT annotation draw controller.
"""
import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.annotation_draw_controller import AnnotationDrawController  # noqa: E402


class _FakeEventPos:
    def __init__(self, x: int, y: int) -> None:
        self._x = x
        self._y = y

    def x(self) -> int:
        return self._x

    def y(self) -> int:
        return self._y


class _FakeEvent:
    def __init__(self, x: int, y: int) -> None:
        self._pos = _FakeEventPos(x, y)

    def position(self) -> _FakeEventPos:
        return self._pos


class _FakeCanvas:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _FakeImageCanvas:
    def __init__(self) -> None:
        self.press_handler = None
        self.sync_count = 0

    def set_mouse_press_handler(self, fn) -> None:
        self.press_handler = fn

    def sync_label_from_canvas(self) -> None:
        self.sync_count += 1


class _FakePainter:
    instances = []

    def __init__(self) -> None:
        self.begin_target = None
        self.pen = None
        self.point_calls = []
        self.rect_calls = []
        self.ended = False
        type(self).instances.append(self)

    def begin(self, target) -> bool:
        self.begin_target = target
        return True

    def setPen(self, pen) -> None:  # noqa: N802
        self.pen = pen

    def drawPoint(self, x: int, y: int) -> None:  # noqa: N802
        self.point_calls.append((x, y))

    def drawRect(self, x: int, y: int, width: int, height: int) -> None:  # noqa: N802
        self.rect_calls.append((x, y, width, height))

    def end(self) -> None:
        self.ended = True


def _make_controller():
    canvas = _FakeCanvas(200, 100)
    image_canvas = _FakeImageCanvas()
    clicked = []
    add_calls = []
    prepared = []
    resets = []
    updates = []
    controller = AnnotationDrawController(
        get_canvas=lambda: canvas,
        image_canvas=image_canvas,
        on_prepare_draw_mode=lambda: prepared.append("prepare"),
        on_clicked_position=lambda x, y: clicked.append((x, y)),
        on_request_add_box=lambda x1, y1, x2, y2: add_calls.append((x1, y1, x2, y2)) or True,
        on_reset_view=lambda: resets.append("reset"),
        on_canvas_updated=lambda: updates.append("updated"),
    )
    return controller, canvas, image_canvas, clicked, add_calls, prepared, resets, updates


def test_draw_controller_enters_draw_mode_and_installs_press_handler() -> None:
    controller, _canvas, image_canvas, _clicked, _add_calls, prepared, _resets, _updates = _make_controller()
    controller._start = (10, 10)

    controller.enter_draw_mode()

    assert prepared == ["prepare"]
    assert image_canvas.press_handler == controller.handle_press
    assert controller._start is None


def test_draw_controller_records_start_point_on_first_click() -> None:
    controller, canvas, image_canvas, clicked, _add_calls, _prepared, _resets, updates = _make_controller()

    with patch("gui.annotation_draw_controller.QPainter", _FakePainter):
        with patch("gui.annotation_draw_controller.QColor", lambda color: color):
            with patch("gui.annotation_draw_controller.QPen", lambda color, width: (color, width)):
                controller.handle_press(_FakeEvent(10, 20))

    assert clicked == [(10, 20)]
    assert image_canvas.sync_count == 1
    assert updates == ["updated"]
    assert _FakePainter.instances[-1].begin_target is canvas
    assert _FakePainter.instances[-1].point_calls == [(10, 20)]


def test_draw_controller_adds_box_after_valid_second_click() -> None:
    controller, canvas, image_canvas, clicked, add_calls, _prepared, resets, updates = _make_controller()

    with patch("gui.annotation_draw_controller.QPainter", _FakePainter):
        with patch("gui.annotation_draw_controller.QColor", lambda color: color):
            with patch("gui.annotation_draw_controller.QPen", lambda color, width: (color, width)):
                controller.handle_press(_FakeEvent(10, 20))
                controller.handle_press(_FakeEvent(50, 70))

    assert clicked == [(10, 20), (50, 70)]
    assert add_calls == [(10, 20, 50, 70)]
    assert resets == []
    assert image_canvas.sync_count == 3
    assert updates == ["updated", "updated", "updated"]
    assert _FakePainter.instances[-1].begin_target is canvas
    assert _FakePainter.instances[-1].rect_calls == [(10, 20, 40, 50)]


def test_draw_controller_resets_view_for_invalid_second_click() -> None:
    controller, _canvas, image_canvas, clicked, add_calls, _prepared, resets, updates = _make_controller()

    with patch("gui.annotation_draw_controller.QPainter", _FakePainter):
        with patch("gui.annotation_draw_controller.QColor", lambda color: color):
            with patch("gui.annotation_draw_controller.QPen", lambda color, width: (color, width)):
                controller.handle_press(_FakeEvent(40, 40))
                controller.handle_press(_FakeEvent(20, 20))

    assert clicked == [(40, 40), (20, 20)]
    assert add_calls == []
    assert resets == ["reset"]
    assert image_canvas.sync_count == 2
    assert updates == ["updated", "updated"]
