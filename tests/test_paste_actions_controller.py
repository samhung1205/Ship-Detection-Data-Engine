"""
Tests for the paste actions controller.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.paste_actions_controller import PasteActionsController  # noqa: E402
from sdde.paste_document import PasteDocument  # noqa: E402


class _FakeItem:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:
        self._text = text


class _FakeListWidget:
    def __init__(self) -> None:
        self._items: list[_FakeItem] = []
        self._current_row = -1

    def addItem(self, text: str) -> None:
        self._items.append(_FakeItem(text))

    def item(self, index: int) -> _FakeItem | None:
        if index < 0 or index >= len(self._items):
            return None
        return self._items[index]

    def takeItem(self, index: int) -> _FakeItem:
        item = self._items.pop(index)
        if not self._items:
            self._current_row = -1
        elif self._current_row >= len(self._items):
            self._current_row = len(self._items) - 1
        return item

    def clear(self) -> None:
        self._items.clear()
        self._current_row = -1

    def currentRow(self) -> int:
        return self._current_row

    def setCurrentRow(self, row: int) -> None:
        self._current_row = row

    def count(self) -> int:
        return len(self._items)


class _FakeLabel:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:
        self.text = text


class _FakeImageCanvas:
    def __init__(self) -> None:
        self.canvas = None
        self.synced = 0

    def set_canvas(self, canvas) -> None:
        self.canvas = canvas

    def sync_label_from_canvas(self) -> None:
        self.synced += 1


def _make_controller(
    *,
    object_names: list[str] | None = None,
    ask_label=None,
    confirm_clear_all=None,
):
    document = PasteDocument()
    list_widget = _FakeListWidget()
    count_label = _FakeLabel()
    image_canvas = _FakeImageCanvas()
    redraw_calls: list[str] = []
    updated: list[str] = []
    object_list = list(object_names or [])
    controller = PasteActionsController(
        parent=None,
        document=document,
        list_widget=list_widget,
        count_label=count_label,
        get_object_names=lambda: list(object_list),
        append_object_name=object_list.append,
        image_canvas=image_canvas,
        on_canvas_updated=lambda: updated.append("updated"),
        on_rows_changed=lambda: redraw_calls.append("redraw"),
        on_add_cancelled=image_canvas.sync_label_from_canvas,
        on_disable_add=lambda: updated.append("disabled"),
        build_record=lambda class_name, real_row: SimpleNamespace(
            class_name=class_name,
            real_row=list(real_row),
        ),
        ask_label=ask_label,
        confirm_clear_all=confirm_clear_all,
    )
    return {
        "controller": controller,
        "document": document,
        "list_widget": list_widget,
        "count_label": count_label,
        "image_canvas": image_canvas,
        "object_list": object_list,
        "redraw_calls": redraw_calls,
        "updated": updated,
    }


def test_paste_actions_controller_adds_candidate_and_record() -> None:
    ctx = _make_controller(
        object_names=["naval", "merchant"],
        ask_label=lambda _parent, _names: ("merchant", True),
    )

    added = ctx["controller"].prompt_add_candidate(
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba", 0.1, 0.2, 0.3, 0.4],
        preview_canvas="preview-canvas",
    )

    assert added is True
    assert ctx["document"].pimg_data == [["merchant", 10, 20, 30, 40, 200, 100]]
    assert ctx["document"].real_pimg_data == [["merchant", 100, 200, 300, 400]]
    assert ctx["document"].paste_images == [["rgba", 0.1, 0.2, 0.3, 0.4]]
    assert ctx["document"].paste_records[0].class_name == "merchant"
    assert ctx["document"].paste_records[0].real_row == ["merchant", 100, 200, 300, 400]
    assert ctx["count_label"].text == "Paste Images  (Total: 1)"
    assert ctx["image_canvas"].canvas == "preview-canvas"
    assert ctx["object_list"] == ["naval", "merchant"]
    assert ctx["updated"] == ["disabled", "updated"]


def test_paste_actions_controller_restores_canvas_when_add_is_cancelled() -> None:
    ctx = _make_controller(
        ask_label=lambda _parent, _names: ("", False),
    )

    added = ctx["controller"].prompt_add_candidate(
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba"],
        preview_canvas="preview-canvas",
    )

    assert added is False
    assert ctx["image_canvas"].synced == 1
    assert ctx["document"].pimg_data == []
    assert ctx["document"].paste_records == []


def test_paste_actions_controller_renames_selected_row_and_record() -> None:
    ctx = _make_controller(
        object_names=["naval", "merchant"],
        ask_label=lambda _parent, _names: ("merchant", True),
    )
    ctx["controller"].add_candidate(
        class_name="naval",
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba"],
        preview_canvas="preview-canvas",
    )
    ctx["list_widget"].setCurrentRow(0)

    ctx["controller"].rename_selected()

    assert ctx["document"].pimg_data[0][0] == "merchant"
    assert ctx["document"].real_pimg_data[0][0] == "merchant"
    assert ctx["document"].paste_records[0].class_name == "merchant"
    assert ctx["list_widget"].item(0).text() == "merchant"
    assert ctx["object_list"] == ["naval", "merchant"]


def test_paste_actions_controller_rejects_unknown_class_names() -> None:
    ctx = _make_controller(object_names=["naval"])

    added = ctx["controller"].add_candidate(
        class_name="merchant",
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba"],
        preview_canvas="preview-canvas",
    )

    assert added is False
    assert ctx["document"].pimg_data == []


def test_paste_actions_controller_rejects_unknown_rename_without_mutating_rows() -> None:
    ctx = _make_controller(
        object_names=["naval"],
        ask_label=lambda _parent, _names: ("merchant", True),
    )
    ctx["controller"].add_candidate(
        class_name="naval",
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba"],
        preview_canvas="preview-canvas",
    )
    ctx["list_widget"].setCurrentRow(0)

    from unittest.mock import patch
    with patch("gui.paste_actions_controller.QtWidgets.QMessageBox.information") as info:
        ctx["controller"].rename_selected()

    info.assert_called_once()
    assert ctx["document"].pimg_data[0][0] == "naval"


def test_paste_actions_controller_removes_and_clears_rows() -> None:
    ctx = _make_controller(
        object_names=["naval", "merchant"],
        confirm_clear_all=lambda _parent: True,
    )
    ctx["controller"].add_candidate(
        class_name="naval",
        bbox_row=[10, 20, 30, 40, 200, 100],
        real_bbox_row=[100, 200, 300, 400],
        paste_image=["rgba-1"],
        preview_canvas="preview-1",
    )
    ctx["controller"].add_candidate(
        class_name="merchant",
        bbox_row=[11, 21, 31, 41, 200, 100],
        real_bbox_row=[101, 201, 301, 401],
        paste_image=["rgba-2"],
        preview_canvas="preview-2",
    )
    ctx["list_widget"].setCurrentRow(0)

    ctx["controller"].delete_selected()

    assert ctx["document"].pimg_data == [["merchant", 11, 21, 31, 41, 200, 100]]
    assert ctx["document"].real_pimg_data == [["merchant", 101, 201, 301, 401]]
    assert len(ctx["document"].paste_images) == 1
    assert len(ctx["document"].paste_records) == 1
    assert ctx["count_label"].text == "Paste Images  (Total: 1)"
    assert ctx["redraw_calls"] == ["redraw"]

    ctx["controller"].clear_all()

    assert ctx["document"].pimg_data == []
    assert ctx["document"].real_pimg_data == []
    assert ctx["document"].paste_images == []
    assert ctx["document"].paste_records == []
    assert ctx["list_widget"].count() == 0
    assert ctx["count_label"].text == "Paste Images  (Total: 0)"
    assert ctx["redraw_calls"] == ["redraw", "redraw"]
