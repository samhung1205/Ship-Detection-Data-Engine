"""
Tests for the paste candidate controller.
"""
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.paste_candidate_controller import PasteCandidateController  # noqa: E402
from sdde.paste_candidate import PasteCandidateSession  # noqa: E402


class _FakePreviewLabel:
    def __init__(self) -> None:
        self.pixmap = None

    def setPixmap(self, pixmap) -> None:
        self.pixmap = pixmap


class _FakeImageCanvas:
    def __init__(self) -> None:
        self.painted = None

    def paint_label_only(self, pixmap) -> None:
        self.painted = pixmap


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

    def scaled(self, width: int, height: int) -> "_FakePixmap":
        return _FakePixmap(width, height)


class _FakeQPixmapFactory:
    @staticmethod
    def fromImage(_image) -> _FakePixmap:  # noqa: N802
        return _FakePixmap(40, 20)


class _FakeQImage:
    class Format:
        Format_RGBA8888 = object()

    def __init__(self, *_args) -> None:
        self.args = _args


class _FakePainter:
    last_instance = None

    def __init__(self) -> None:
        self.begin_target = None
        self.draw_calls = []
        self.ended = False
        type(self).last_instance = self

    def begin(self, target) -> bool:
        self.begin_target = target
        return True

    def drawImage(self, rect, image) -> None:  # noqa: N802
        self.draw_calls.append((rect, image))

    def end(self) -> None:
        self.ended = True


class _FakeEvent:
    class _Pos:
        def __init__(self, x: int, y: int) -> None:
            self._x = x
            self._y = y

        def x(self) -> int:
            return self._x

        def y(self) -> int:
            return self._y

    def __init__(self, x: int, y: int) -> None:
        self._pos = self._Pos(x, y)

    def position(self):
        return self._pos


def _make_controller():
    session = PasteCandidateSession()
    preview_label = _FakePreviewLabel()
    image_canvas = _FakeImageCanvas()
    pressed = []
    prepared = []
    enabled = []
    labels = []
    updates = []
    handler_holder = {}
    controller = PasteCandidateController(
        session=session,
        get_canvas=lambda: _FakePixmap(200, 100),
        get_origin_size=lambda: (1000, 500),
        image_canvas=image_canvas,
        preview_label=preview_label,
        set_mouse_press_handler=lambda fn: handler_holder.setdefault("handler", fn),
        get_adjustments=lambda: (50, 0, 100, 100),
        on_prepare_paste_mode=lambda: prepared.append("prepared"),
        on_clicked_position=lambda x, y: pressed.append((x, y)),
        on_enable_add=lambda enabled_flag: enabled.append(enabled_flag),
        on_set_adjustment_labels=lambda a, b, c, d: labels.append((a, b, c, d)),
        on_canvas_updated=lambda: updates.append("updated"),
    )
    return {
        "controller": controller,
        "session": session,
        "preview_label": preview_label,
        "image_canvas": image_canvas,
        "pressed": pressed,
        "prepared": prepared,
        "enabled": enabled,
        "labels": labels,
        "updates": updates,
        "handler_holder": handler_holder,
    }


def test_candidate_controller_loads_asset_and_refreshes_thumbnail() -> None:
    ctx = _make_controller()
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    rgba[1:3, 1:3, 3] = 255

    with patch("gui.paste_candidate_controller.cv2.imread", return_value=rgba):
        with patch.object(ctx["controller"], "_refresh_thumbnail") as refresh:
            loaded = ctx["controller"].load_asset("/tmp/ship.png")

    assert loaded is True
    assert ctx["session"].asset_path == "/tmp/ship.png"
    assert ctx["session"].pasteimg is not None
    assert ctx["session"].origin_pasteimg is not None
    assert ctx["session"].has_anchor is False
    refresh.assert_called_once()


def test_candidate_controller_enters_paste_mode_and_sets_handler() -> None:
    ctx = _make_controller()

    ctx["controller"].enter_paste_mode()

    assert ctx["prepared"] == ["prepared"]
    assert callable(ctx["handler_holder"]["handler"])


def test_candidate_controller_handle_press_sets_anchor_and_recomputes() -> None:
    ctx = _make_controller()
    recomputed = []
    ctx["controller"].recompute_preview = lambda: recomputed.append("preview")

    ctx["controller"].handle_press(_FakeEvent(50, 30))

    assert ctx["pressed"] == [(50, 30)]
    assert ctx["prepared"] == ["prepared"]
    assert ctx["enabled"] == [True]
    assert ctx["session"].anchor_x == 50
    assert ctx["session"].anchor_y == 30
    assert recomputed == ["preview"]


def test_candidate_controller_horizontal_flip_refreshes_and_recomputes_when_anchored() -> None:
    ctx = _make_controller()
    ctx["session"].origin_pasteimg = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    ctx["session"].pasteimg = ctx["session"].origin_pasteimg.copy()
    ctx["session"].set_anchor(20, 10)
    recomputed = []

    with patch.object(ctx["controller"], "_refresh_thumbnail") as refresh:
        ctx["controller"].recompute_preview = lambda: recomputed.append("preview")
        flipped = ctx["controller"].set_horizontal_flip(True)

    assert flipped is True
    assert np.array_equal(
        ctx["session"].pasteimg,
        ctx["session"].origin_pasteimg[:, ::-1, :],
    )
    refresh.assert_called_once()
    assert recomputed == ["preview"]


def test_candidate_controller_recompute_preview_populates_candidate_state() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.pasteimg_width = 3
    session.pasteimg_height = 3
    session.pasteimg_channel = 4
    session.set_anchor(50, 30)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert ctx["labels"] == [("100 %", "0 °", "100", "100")]
    assert session.norm_pimg is not None
    assert session.bbox_pimg is not None
    assert session.real_bbox_pimg is not None
    assert ctx["image_canvas"].painted is not None
    assert ctx["updates"] == ["updated"]
