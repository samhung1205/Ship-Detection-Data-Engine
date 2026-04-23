"""
Controller for GT annotation list interactions.

This keeps widget-event orchestration out of ``main_window.py`` while the
legacy Qt widgets are still in place.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction


class AnnotationListController:
    def __init__(
        self,
        *,
        parent: Any,
        list_view: Any,
        get_object_names: Callable[[], Sequence[str]],
        on_preview_row: Callable[[int], None],
        on_row_changed: Callable[[int], None],
        on_delete_row: Callable[[int], None],
        on_rename_row: Callable[[int, str, str], None],
        on_clear_all: Callable[[], None],
    ) -> None:
        self._parent = parent
        self._list_view = list_view
        self._get_object_names = get_object_names
        self._on_preview_row = on_preview_row
        self._on_row_changed = on_row_changed
        self._on_delete_row = on_delete_row
        self._on_rename_row = on_rename_row
        self._on_clear_all = on_clear_all

    def on_row_clicked(self, *_args: object) -> None:
        row = self._list_view.current_index_row()
        if row >= 0:
            self._on_preview_row(row)

    def on_row_changed(self, row: int) -> None:
        self._on_row_changed(row)

    def delete_selected(self) -> None:
        row = self._list_view.current_row()
        if row < 0:
            return
        self._on_delete_row(row)

    def rename_selected(self) -> None:
        row = self._list_view.current_row()
        if row < 0:
            return
        names = list(self._get_object_names())
        text, ok = QtWidgets.QInputDialog().getItem(
            self._parent, '', 'Enter object name', names, 0, False
        )
        if not ok:
            return
        old_name = self._list_view.item_text(row)
        if old_name == text:
            return
        self._on_rename_row(row, old_name, text)

    def confirm_clear_all(self) -> None:
        ret = QtWidgets.QMessageBox.question(
            self._parent,
            'question',
            'Delete all?',
            QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            self._on_clear_all()

    def open_context_menu(self, pos) -> None:
        if self._list_view.current_index_row() < 0:
            return
        context = QtWidgets.QMenu(self._parent)
        action_rename = QAction("Rename", self._parent)
        action_delete = QAction("Delete", self._parent)
        context.addAction(action_rename)
        context.addAction(action_delete)
        action_rename.triggered.connect(self.rename_selected)
        action_delete.triggered.connect(self.delete_selected)
        context.exec(self._list_view.viewport_global_pos(pos))
