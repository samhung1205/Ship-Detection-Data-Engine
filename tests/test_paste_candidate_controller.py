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
from sdde.augmentation import PasteAdjustments  # noqa: E402
from sdde.paste_candidate import PasteCandidateSession  # noqa: E402


class _FakePreviewLabel:
    def __init__(self) -> None:
        self.pixmap = None
        self._width = 100
        self._height = 120

    def setPixmap(self, pixmap) -> None:
        self.pixmap = pixmap

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


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

    def scaled(self, width: int, height: int, *_args) -> "_FakePixmap":
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

    def copy(self):
        return self


class _FakePainter:
    last_instance = None
    class RenderHint:
        SmoothPixmapTransform = object()

    def __init__(self) -> None:
        self.begin_target = None
        self.draw_calls = []
        self.ended = False
        type(self).last_instance = self

    def begin(self, target) -> bool:
        self.begin_target = target
        return True

    def setRenderHint(self, *_args) -> None:  # noqa: N802
        return None

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
    statuses = []
    handler_holder = {}
    adjustments_holder = {"value": PasteAdjustments()}
    smart_mode_holder = {"enabled": False}
    controller = PasteCandidateController(
        session=session,
        get_canvas=lambda: _FakePixmap(200, 100),
        get_origin_size=lambda: (1000, 500),
        image_canvas=image_canvas,
        preview_label=preview_label,
        set_mouse_press_handler=lambda fn: handler_holder.setdefault("handler", fn),
        get_adjustments=lambda: adjustments_holder["value"],
        is_smart_mode_enabled=lambda: smart_mode_holder["enabled"],
        get_smart_zone_rect=lambda: session.smart_zone_rect,
        on_prepare_paste_mode=lambda: prepared.append("prepared"),
        on_clicked_position=lambda x, y: pressed.append((x, y)),
        on_enable_add=lambda enabled_flag: enabled.append(enabled_flag),
        on_set_adjustment_labels=lambda adjustments: labels.append(adjustments),
        on_set_status_message=lambda message: statuses.append(message),
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
        "statuses": statuses,
        "handler_holder": handler_holder,
        "adjustments_holder": adjustments_holder,
        "smart_mode_holder": smart_mode_holder,
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
    assert ctx["enabled"] == []
    assert ctx["session"].anchor_x == 50
    assert ctx["session"].anchor_y == 30
    assert recomputed == ["preview"]


def test_candidate_controller_flip_preview_is_rebuilt_from_origin() -> None:
    ctx = _make_controller()
    origin = np.arange(24, dtype=np.uint8).reshape((2, 3, 4))
    ctx["session"].origin_pasteimg = origin.copy()

    with patch.object(ctx["controller"], "_refresh_thumbnail") as refresh:
        ctx["adjustments_holder"]["value"] = PasteAdjustments(h_flip=True)
        assert ctx["controller"].set_horizontal_flip(True) is True
        assert np.array_equal(ctx["session"].pasteimg, origin[:, ::-1, :])

        ctx["adjustments_holder"]["value"] = PasteAdjustments(v_flip=True)
        assert ctx["controller"].set_vertical_flip(True) is True
        assert np.array_equal(ctx["session"].pasteimg, origin[::-1, :, :])

    assert refresh.call_count == 2


def test_candidate_controller_recompute_preview_populates_candidate_state() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert len(ctx["labels"]) == 1
    assert ctx["labels"][0].opacity_pct == 100
    assert session.norm_pimg is not None
    assert len(session.norm_pimg) == 10
    assert session.bbox_pimg is not None
    assert session.real_bbox_pimg is not None
    assert ctx["image_canvas"].painted is not None
    assert ctx["updates"] == ["updated"]
    assert ctx["enabled"][-1] is True


def test_candidate_controller_builds_origin_resolution_export_payload() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert session.norm_pimg is not None
    assert session.norm_pimg[5] is not None
    assert session.norm_pimg[8] == 15 / 1000
    assert session.norm_pimg[9] == 15 / 500
    assert session.real_bbox_pimg == [243, 143, 258, 158]


def test_candidate_controller_disables_add_when_alpha_becomes_empty() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)
    ctx["adjustments_holder"]["value"] = PasteAdjustments(opacity_pct=0)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert session.norm_pimg is None
    assert session.bbox_pimg is None
    assert session.real_bbox_pimg is None
    assert ctx["enabled"][-1] is False


def test_candidate_controller_smart_mode_requires_zone() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)
    ctx["smart_mode_holder"]["enabled"] = True

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert session.real_bbox_pimg is not None
    assert ctx["enabled"][-1] is False
    assert "Smart zone mode is on" in ctx["statuses"][-1]


def test_candidate_controller_smart_mode_accepts_bbox_inside_zone() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)
    session.smart_zone_rect = (200, 100, 300, 200)
    ctx["smart_mode_holder"]["enabled"] = True

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()

    assert ctx["enabled"][-1] is True
    assert "inside the smart zone" in ctx["statuses"][-1]


def test_candidate_controller_shadow_does_not_expand_bbox() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((3, 3, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()
                baseline_bbox = list(session.real_bbox_pimg)
                ctx["adjustments_holder"]["value"] = PasteAdjustments(
                    shadow_enabled=True,
                    shadow_opacity_pct=60,
                    shadow_offset_px=12,
                )
                ctx["controller"].recompute_preview()

    assert session.real_bbox_pimg == baseline_bbox
    assert ctx["enabled"][-1] is True


def test_candidate_controller_motion_blur_keeps_bbox_and_add_valid() -> None:
    ctx = _make_controller()
    session = ctx["session"]
    session.origin_pasteimg = np.ones((5, 5, 4), dtype=np.uint8) * 255
    session.set_anchor(50, 30)

    with patch("gui.paste_candidate_controller.QPainter", _FakePainter):
        with patch("gui.paste_candidate_controller.QImage", _FakeQImage):
            with patch("gui.paste_candidate_controller.QPixmap", _FakeQPixmapFactory):
                ctx["controller"].recompute_preview()
                baseline_bbox = list(session.real_bbox_pimg)
                ctx["adjustments_holder"]["value"] = PasteAdjustments(
                    motion_blur_enabled=True,
                    motion_blur_length=13,
                    motion_blur_angle_deg=30,
                )
                ctx["controller"].recompute_preview()

    assert session.real_bbox_pimg == baseline_bbox
    assert ctx["enabled"][-1] is True
