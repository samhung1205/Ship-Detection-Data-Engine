"""
Tests for the GT annotation list controller.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.annotation_list_controller import AnnotationListController  # noqa: E402


class _FakeListView:
    def __init__(self) -> None:
        self._current_row = -1
        self._items: list[str] = []

    def set_current_row(self, row: int) -> None:
        self._current_row = row

    def current_row(self) -> int:
        return self._current_row

    def current_index_row(self) -> int:
        return self._current_row

    def item_text(self, index: int) -> str:
        return self._items[index]


def test_annotation_list_controller_previews_clicked_row() -> None:
    called: list[int] = []
    list_view = _FakeListView()
    list_view._items = ["naval"]
    list_view.set_current_row(0)
    controller = AnnotationListController(
        parent=None,
        list_view=list_view,
        get_object_names=lambda: ["naval"],
        on_preview_row=called.append,
        on_row_changed=lambda _row: None,
        on_delete_row=lambda _row: None,
        on_rename_row=lambda _row, _old, _new: None,
        on_clear_all=lambda: None,
    )

    controller.on_row_clicked()

    assert called == [0]


def test_annotation_list_controller_forwards_row_changed() -> None:
    called: list[int] = []
    controller = AnnotationListController(
        parent=None,
        list_view=_FakeListView(),
        get_object_names=lambda: [],
        on_preview_row=lambda _row: None,
        on_row_changed=called.append,
        on_delete_row=lambda _row: None,
        on_rename_row=lambda _row, _old, _new: None,
        on_clear_all=lambda: None,
    )

    controller.on_row_changed(3)

    assert called == [3]


def test_annotation_list_controller_deletes_selected_row() -> None:
    called: list[int] = []
    list_view = _FakeListView()
    list_view._items = ["naval", "merchant"]
    list_view.set_current_row(1)
    controller = AnnotationListController(
        parent=None,
        list_view=list_view,
        get_object_names=lambda: [],
        on_preview_row=lambda _row: None,
        on_row_changed=lambda _row: None,
        on_delete_row=called.append,
        on_rename_row=lambda _row, _old, _new: None,
        on_clear_all=lambda: None,
    )

    controller.delete_selected()

    assert called == [1]
