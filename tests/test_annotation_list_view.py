"""
Tests for the GT annotation list view adapter.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.annotation_list_view import AnnotationListView  # noqa: E402


class _FakeItem:
    def __init__(self, text: str) -> None:
        self._text = text

    def text(self) -> str:
        return self._text

    def setText(self, text: str) -> None:  # noqa: N802
        self._text = text


class _FakeList:
    def __init__(self) -> None:
        self._items: list[_FakeItem] = []
        self._current_row = -1

    def addItem(self, text: str) -> None:  # noqa: N802
        self._items.append(_FakeItem(text))

    def insertItem(self, index: int, text: str) -> None:  # noqa: N802
        self._items.insert(index, _FakeItem(text))

    def takeItem(self, index: int) -> None:  # noqa: N802
        self._items.pop(index)

    def item(self, index: int) -> _FakeItem:  # noqa: N802
        return self._items[index]

    def clear(self) -> None:
        self._items.clear()
        self._current_row = -1

    def count(self) -> int:
        return len(self._items)

    def setCurrentRow(self, row: int) -> None:  # noqa: N802
        self._current_row = row

    def currentRow(self) -> int:  # noqa: N802
        return self._current_row

    def currentIndex(self):  # noqa: N802
        class _Idx:
            def __init__(self, row: int) -> None:
                self._row = row

            def row(self) -> int:
                return self._row

        return _Idx(self._current_row)


class _Label:
    def __init__(self) -> None:
        self.text = ""

    def setText(self, text: str) -> None:  # noqa: N802
        self.text = text


def test_annotation_list_view_updates_total_and_items() -> None:
    label = _Label()
    list_widget = _FakeList()
    view = AnnotationListView(count_label=label, list_widget=list_widget)

    view.sync_labels(["naval", "merchant"])

    assert label.text == "Box Labels  (Total: 2)"
    assert list_widget.count() == 2
    assert view.item_text(1) == "merchant"


def test_annotation_list_view_supports_add_remove_rename() -> None:
    view = AnnotationListView(count_label=_Label(), list_widget=_FakeList())
    view.add_item("naval")
    view.add_item("merchant")

    view.rename_item(1, "dock")
    view.remove_item(0)

    assert view.count() == 1
    assert view.item_text(0) == "dock"


def test_annotation_list_view_tracks_current_row() -> None:
    view = AnnotationListView(count_label=_Label(), list_widget=_FakeList())
    view.add_item("naval")
    view.set_current_row(0)

    assert view.current_row() == 0
    assert view.current_index_row() == 0
