"""
Unit tests for annotation undo/redo command objects (no PyQt widgets).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.document import AnnotationDocument  # noqa: E402
from gui.annotation_list_view import AnnotationListView  # noqa: E402
from gui.annotation_controller import (  # noqa: E402
    AddBoxCommand,
    AnnotationController,
    BulkAppendBoxesCommand,
    ClearAllBoxesCommand,
    MAX_UNDO,
    RemoveBoxCommand,
    RenameBoxCommand,
    ReplaceAllBoxesCommand,
    UpdateBoxGeometryCommand,
)


class _FakeItem:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, t: str) -> None:  # noqa: N802
        self._text = t


class _FakeList:
    def __init__(self) -> None:
        self._items: list[_FakeItem] = []

    def count(self) -> int:
        return len(self._items)

    def addItem(self, text: str) -> None:  # noqa: N802
        self._items.append(_FakeItem(text))

    def takeItem(self, i: int) -> None:  # noqa: N802
        self._items.pop(i)

    def insertItem(self, i: int, text: str) -> None:  # noqa: N802
        self._items.insert(i, _FakeItem(text))

    def item(self, i: int) -> _FakeItem:  # noqa: N802
        return self._items[i]

    def clear(self) -> None:
        self._items.clear()


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, s: str) -> None:  # noqa: N802
        self.text = s


class _HideBox:
    def __init__(self) -> None:
        self._checked = False

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, v: bool) -> None:  # noqa: N802
        self._checked = v


class _FakeW:
    """Minimal surface used by gui.annotation_controller._refresh_canvas."""

    def __init__(self) -> None:
        self.gt_document = AnnotationDocument()
        self.object_list: list = []
        self.listwidget = _FakeList()
        self.label_list = _Label()
        self.gt_list_view = AnnotationListView(
            count_label=self.label_list,
            list_widget=self.listwidget,
        )
        self.hideBox = _HideBox()
        self.set_img_ratio_calls = 0

    def set_img_ratio(self) -> None:
        self.set_img_ratio_calls += 1

    @property
    def data(self) -> list:
        return self.gt_document.data

    @property
    def real_data(self) -> list:
        return self.gt_document.real_data

    @property
    def box_attributes(self) -> list:
        return self.gt_document.box_attributes


def _apply_add(
    controller: AnnotationController,
    name: str,
    *,
    canvas_w: int = 100,
    canvas_h: int = 100,
    real_x1: int = 10,
    real_y1: int = 10,
    real_x2: int = 20,
    real_y2: int = 20,
    extended_object_list: bool = False,
) -> None:
    controller.apply(
        AddBoxCommand(
            [name, 1, 1, 2, 2, canvas_w, canvas_h],
            [name, real_x1, real_y1, real_x2, real_y2],
            name,
            extended_object_list=extended_object_list,
        )
    )


def test_add_undo_redo() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "a", extended_object_list=True)
    assert len(w.data) == 1
    assert w.object_list == []
    c.undo()
    assert len(w.data) == 0
    assert len(w.box_attributes) == 0
    assert w.object_list == []
    c.redo()
    assert len(w.data) == 1
    assert len(w.box_attributes) == 1


def test_undo_stack_cap() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    for i in range(MAX_UNDO + 3):
        _apply_add(
            c,
            f"x{i}",
            canvas_w=1,
            canvas_h=1,
            real_x1=0,
            real_y1=0,
            real_x2=1,
            real_y2=1,
            extended_object_list=False,
        )
    assert len(c._undo) == MAX_UNDO


def test_remove_box_undo_redo_keeps_parallel_state_aligned() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "naval", extended_object_list=True)
    _apply_add(c, "merchant", extended_object_list=True)

    c.apply(RemoveBoxCommand(0))

    assert [row[0] for row in w.real_data] == ["merchant"]
    assert w.listwidget.count() == 1
    assert w.listwidget.item(0).text() == "merchant"
    assert len(w.box_attributes) == 1
    assert w.label_list.text == "Box Labels  (Total: 1)"

    c.undo()

    assert [row[0] for row in w.real_data] == ["naval", "merchant"]
    assert w.listwidget.count() == 2
    assert [w.listwidget.item(i).text() for i in range(2)] == ["naval", "merchant"]
    assert len(w.box_attributes) == 2
    assert w.label_list.text == "Box Labels  (Total: 2)"

    c.redo()

    assert [row[0] for row in w.real_data] == ["merchant"]
    assert w.listwidget.count() == 1
    assert len(w.box_attributes) == 1


def test_rename_box_undo_redo_updates_object_list_and_labels() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    w.object_list = ["naval", "merchant"]
    _apply_add(c, "naval", extended_object_list=False)

    c.apply(RenameBoxCommand(0, "naval", "merchant"))

    assert w.data[0][0] == "merchant"
    assert w.real_data[0][0] == "merchant"
    assert w.listwidget.item(0).text() == "merchant"
    assert w.object_list == ["naval", "merchant"]

    c.undo()

    assert w.data[0][0] == "naval"
    assert w.real_data[0][0] == "naval"
    assert w.listwidget.item(0).text() == "naval"
    assert w.object_list == ["naval", "merchant"]


def test_update_box_geometry_undo_redo_restores_previous_rows() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "naval", extended_object_list=True)
    w.gt_document.set_box_attributes(0, {"crowded": "true", "size_tag": "large"})

    c.apply(
        UpdateBoxGeometryCommand(
            0,
            ["naval", 5, 6, 25, 30, 100, 100],
            ["naval", 50, 60, 250, 300],
        )
    )

    assert w.data[0] == ["naval", 5, 6, 25, 30, 100, 100]
    assert w.real_data[0] == ["naval", 50, 60, 250, 300]
    assert w.box_attributes[0]["crowded"] == "true"
    assert w.box_attributes[0]["size_tag"] == "large"

    c.undo()

    assert w.data[0] == ["naval", 1, 1, 2, 2, 100, 100]
    assert w.real_data[0] == ["naval", 10, 10, 20, 20]
    assert w.box_attributes[0]["crowded"] == "true"
    assert w.box_attributes[0]["size_tag"] == "large"

    c.redo()


def test_replace_all_boxes_command_restores_previous_snapshot_on_undo() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "naval", extended_object_list=True)
    _apply_add(c, "merchant", extended_object_list=True)

    c.apply(
        ReplaceAllBoxesCommand(
            [
                (
                    ["dock", 5, 5, 6, 6, 100, 100],
                    ["dock", 50, 50, 60, 60],
                    "dock",
                )
            ]
        )
    )

    assert [row[0] for row in w.real_data] == ["dock"]
    assert w.listwidget.count() == 1
    assert w.listwidget.item(0).text() == "dock"

    c.undo()

    assert [row[0] for row in w.real_data] == ["naval", "merchant"]
    assert [w.listwidget.item(i).text() for i in range(2)] == ["naval", "merchant"]

    c.redo()

    assert [row[0] for row in w.real_data] == ["dock"]


def test_replace_all_boxes_accepts_empty_import_and_can_undo() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "naval", extended_object_list=True)

    c.replace_all_boxes([])

    assert w.real_data == []
    assert w.listwidget.count() == 0
    assert w.label_list.text == "Box Labels  (Total: 0)"

    c.undo()

    assert [row[0] for row in w.real_data] == ["naval"]


def test_clear_all_undo_redo_restores_boxes_labels_and_attributes() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    _apply_add(c, "naval", extended_object_list=True)
    _apply_add(c, "merchant", extended_object_list=True)

    c.apply(ClearAllBoxesCommand())

    assert w.data == []
    assert w.real_data == []
    assert w.listwidget.count() == 0
    assert w.box_attributes == []
    assert w.label_list.text == "Box Labels  (Total: 0)"

    c.undo()

    assert [row[0] for row in w.real_data] == ["naval", "merchant"]
    assert [w.listwidget.item(i).text() for i in range(2)] == ["naval", "merchant"]
    assert len(w.box_attributes) == 2
    assert w.label_list.text == "Box Labels  (Total: 2)"

    c.redo()

    assert w.real_data == []
    assert w.listwidget.count() == 0
    assert w.box_attributes == []


def test_bulk_append_undo_redo_keeps_attribute_alignment() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    blocks = [
        (
            ["naval", 1, 1, 2, 2, 100, 100],
            ["naval", 10, 10, 20, 20],
            "naval",
        ),
        (
            ["merchant", 2, 2, 3, 3, 100, 100],
            ["merchant", 30, 30, 40, 40],
            "merchant",
        ),
    ]

    c.apply(BulkAppendBoxesCommand(blocks))

    assert [row[0] for row in w.real_data] == ["naval", "merchant"]
    assert w.listwidget.count() == 2
    assert len(w.box_attributes) == 2
    assert w.label_list.text == "Box Labels  (Total: 2)"

    c.undo()

    assert w.real_data == []
    assert w.listwidget.count() == 0
    assert w.box_attributes == []
    assert w.label_list.text == "Box Labels  (Total: 0)"

    c.redo()

    assert [row[0] for row in w.real_data] == ["naval", "merchant"]
    assert w.listwidget.count() == 2
    assert len(w.box_attributes) == 2


def test_controller_helper_methods_route_to_command_objects() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    seen = []
    c.apply = lambda cmd: seen.append(type(cmd).__name__)  # type: ignore[method-assign]

    c.add_box(
        ["naval", 1, 1, 2, 2, 100, 100],
        ["naval", 10, 10, 20, 20],
        "naval",
        extended_object_list=True,
    )
    c.remove_box(0)
    c.rename_box(0, "naval", "merchant")
    c.update_box_geometry(
        0,
        ["naval", 5, 5, 15, 15, 100, 100],
        ["naval", 50, 50, 150, 150],
    )
    c.clear_all_boxes()
    c.append_blocks(
        [(["naval", 1, 1, 2, 2, 100, 100], ["naval", 10, 10, 20, 20], "naval")]
    )
    c.replace_all_boxes(
        [(["dock", 3, 3, 4, 4, 100, 100], ["dock", 30, 30, 40, 40], "dock")]
    )

    assert seen == [
        "AddBoxCommand",
        "RemoveBoxCommand",
        "RenameBoxCommand",
        "UpdateBoxGeometryCommand",
        "ClearAllBoxesCommand",
        "BulkAppendBoxesCommand",
        "ReplaceAllBoxesCommand",
    ]


if __name__ == "__main__":
    test_add_undo_redo()
    test_undo_stack_cap()
    test_remove_box_undo_redo_keeps_parallel_state_aligned()
    test_rename_box_undo_redo_updates_object_list_and_labels()
    test_clear_all_undo_redo_restores_boxes_labels_and_attributes()
    test_bulk_append_undo_redo_keeps_attribute_alignment()
    print("OK")
