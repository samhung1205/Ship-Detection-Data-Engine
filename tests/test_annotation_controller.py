"""
Unit tests for annotation undo/redo command objects (no PyQt widgets).
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.annotation_controller import (  # noqa: E402
    AddBoxCommand,
    AnnotationController,
    MAX_UNDO,
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
        self.data: list = []
        self.real_data: list = []
        self.object_list: list = []
        self.listwidget = _FakeList()
        self.label_list = _Label()
        self.hideBox = _HideBox()
        self.set_img_ratio_calls = 0
        self.box_attributes: list = []

    def append_box_attributes_row(self) -> None:
        self.box_attributes.append(
            {
                "size_tag": "medium",
                "crowded": "false",
                "difficulty_tag": "normal",
                "scene_tag": "unknown",
            }
        )

    def set_img_ratio(self) -> None:
        self.set_img_ratio_calls += 1


def test_add_undo_redo() -> None:
    w = _FakeW()
    c = AnnotationController(w)
    c.apply(
        AddBoxCommand(
            ["a", 1, 1, 2, 2, 100, 100],
            ["a", 10, 10, 20, 20],
            "a",
            extended_object_list=True,
        )
    )
    assert len(w.data) == 1
    assert w.object_list == ["a"]
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
        c.apply(
            AddBoxCommand(
                [f"x{i}", 0, 0, 1, 1, 1, 1],
                [f"x{i}", 0, 0, 1, 1],
                f"x{i}",
                extended_object_list=False,
            )
        )
    assert len(c._undo) == MAX_UNDO


if __name__ == "__main__":
    test_add_undo_redo()
    test_undo_stack_cap()
    print("OK")
