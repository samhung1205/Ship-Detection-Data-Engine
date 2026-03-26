"""
Tests for direct drag / resize editing of GT boxes.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.document import AnnotationDocument  # noqa: E402
from gui.annotation_edit_controller import AnnotationEditController  # noqa: E402


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
        self.sync_count = 0

    def sync_label_from_canvas(self) -> None:
        self.sync_count += 1


class _FakeListView:
    def __init__(self, row: int) -> None:
        self._row = row

    def current_row(self) -> int:
        return self._row


def _make_controller():
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 10, 20, 30, 50, 100, 100],
        ["naval", 100, 200, 300, 500],
    )
    canvas = _FakeCanvas(200, 100)
    image_canvas = _FakeImageCanvas()
    render_calls: list[int | None] = []
    update_calls: list[tuple[int, int, int, int, int]] = []
    preview_calls: list[int] = []
    updates: list[str] = []
    controller = AnnotationEditController(
        document=doc,
        list_view=_FakeListView(0),
        get_canvas=lambda: canvas,
        image_canvas=image_canvas,
        render_canvas=lambda exclude_row: render_calls.append(exclude_row),
        on_request_update_box=lambda row, x1, y1, x2, y2: update_calls.append((row, x1, y1, x2, y2)) or True,
        on_restore_preview=lambda row: preview_calls.append(row),
        on_canvas_updated=lambda: updates.append("updated"),
    )
    return controller, image_canvas, render_calls, update_calls, preview_calls, updates


def test_edit_controller_moves_selected_box_and_commits_on_release() -> None:
    controller, image_canvas, render_calls, update_calls, preview_calls, updates = _make_controller()
    import gui.annotation_edit_controller as edit_module

    overlays: list[tuple[int, int, int, int]] = []
    orig_overlay = edit_module.draw_selection_overlay
    try:
        edit_module.draw_selection_overlay = lambda _canvas, *, x1, y1, x2, y2, fill_color, outline_color: overlays.append((x1, y1, x2, y2))  # type: ignore[assignment]
        assert controller.handle_press(_FakeEvent(30, 30)) is True
        assert controller.handle_move(_FakeEvent(50, 45)) is True
        assert controller.handle_release(_FakeEvent(50, 45)) is True
    finally:
        edit_module.draw_selection_overlay = orig_overlay  # type: ignore[assignment]

    assert render_calls == [0, 0]
    assert overlays == [(20, 20, 60, 50), (40, 35, 80, 65)]
    assert image_canvas.sync_count == 2
    assert update_calls == [(0, 40, 35, 80, 65)]
    assert preview_calls == [0]
    assert updates == ["updated", "updated"]


def test_edit_controller_resizes_from_corner_handle() -> None:
    controller, _image_canvas, render_calls, update_calls, preview_calls, _updates = _make_controller()
    import gui.annotation_edit_controller as edit_module

    overlays: list[tuple[int, int, int, int]] = []
    orig_overlay = edit_module.draw_selection_overlay
    try:
        edit_module.draw_selection_overlay = lambda _canvas, *, x1, y1, x2, y2, fill_color, outline_color: overlays.append((x1, y1, x2, y2))  # type: ignore[assignment]
        assert controller.handle_press(_FakeEvent(60, 50)) is True
        assert controller.handle_move(_FakeEvent(90, 70)) is True
        assert controller.handle_release(_FakeEvent(90, 70)) is True
    finally:
        edit_module.draw_selection_overlay = orig_overlay  # type: ignore[assignment]

    assert render_calls == [0, 0]
    assert overlays[-1] == (20, 20, 90, 70)
    assert update_calls == [(0, 20, 20, 90, 70)]
    assert preview_calls == [0]


def test_edit_controller_ignores_press_outside_selected_box() -> None:
    controller, image_canvas, render_calls, update_calls, preview_calls, updates = _make_controller()

    assert controller.handle_press(_FakeEvent(5, 5)) is False
    assert controller.handle_move(_FakeEvent(10, 10)) is False
    assert controller.handle_release(_FakeEvent(10, 10)) is False
    assert image_canvas.sync_count == 0
    assert render_calls == []
    assert update_calls == []
    assert preview_calls == []
    assert updates == []
