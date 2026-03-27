"""
Controller for committed paste-image actions.

This keeps paste add / rename / delete / clear flows out of ``main_window.py``
while the candidate transform and preview math still live in the widget.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction


AskLabelFn = Callable[[Any, Sequence[str]], tuple[str, bool]]
ConfirmClearFn = Callable[[Any], bool]
BuildRecordFn = Callable[[str, list[object]], Any | None]


def _default_ask_label(parent: Any, names: Sequence[str]) -> tuple[str, bool]:
    return QtWidgets.QInputDialog().getItem(
        parent,
        "",
        "Enter object name",
        list(names),
        0,
    )


def _default_confirm_clear(parent: Any) -> bool:
    ret = QtWidgets.QMessageBox.question(
        parent,
        "question",
        "Delete all?",
        QtWidgets.QMessageBox.StandardButton.Cancel,
        QtWidgets.QMessageBox.StandardButton.Ok,
    )
    return ret == QtWidgets.QMessageBox.StandardButton.Ok


class PasteActionsController:
    def __init__(
        self,
        *,
        parent: Any,
        document: Any,
        list_widget: Any,
        count_label: Any,
        get_object_names: Callable[[], Sequence[str]],
        append_object_name: Callable[[str], None],
        image_canvas: Any,
        on_canvas_updated: Callable[[], None],
        on_rows_changed: Callable[[], None] | None = None,
        on_add_cancelled: Callable[[], None] | None = None,
        on_disable_add: Callable[[], None] | None = None,
        build_record: BuildRecordFn | None = None,
        ask_label: AskLabelFn | None = None,
        confirm_clear_all: ConfirmClearFn | None = None,
    ) -> None:
        self._parent = parent
        self._document = document
        self._list_widget = list_widget
        self._count_label = count_label
        self._get_object_names = get_object_names
        self._append_object_name = append_object_name
        self._image_canvas = image_canvas
        self._on_canvas_updated = on_canvas_updated
        self._on_rows_changed = on_rows_changed
        self._on_add_cancelled = on_add_cancelled
        self._on_disable_add = on_disable_add
        self._build_record = build_record
        self._ask_label = ask_label or _default_ask_label
        self._confirm_clear_all = confirm_clear_all or _default_confirm_clear

    def current_row(self) -> int:
        return int(self._list_widget.currentRow())

    def prompt_add_candidate(
        self,
        *,
        bbox_row: Sequence[object] | None,
        real_bbox_row: Sequence[object] | None,
        paste_image: Any,
        preview_canvas: Any,
    ) -> bool:
        if (
            bbox_row is None
            or real_bbox_row is None
            or paste_image is None
            or preview_canvas is None
            or len(bbox_row) < 6
            or len(real_bbox_row) < 4
        ):
            return False
        item, ok = self._ask_label(self._parent, list(self._get_object_names()))
        if not ok:
            if self._on_add_cancelled is not None:
                self._on_add_cancelled()
            return False
        return self.add_candidate(
            class_name=item,
            bbox_row=bbox_row,
            real_bbox_row=real_bbox_row,
            paste_image=paste_image,
            preview_canvas=preview_canvas,
        )

    def add_candidate(
        self,
        *,
        class_name: str,
        bbox_row: Sequence[object],
        real_bbox_row: Sequence[object],
        paste_image: Any,
        preview_canvas: Any,
    ) -> bool:
        data_row = [class_name, *list(bbox_row)]
        real_row = [class_name, *list(real_bbox_row)]
        self._list_widget.addItem(class_name)
        if class_name not in self._get_object_names():
            self._append_object_name(class_name)
        record = None
        if self._build_record is not None:
            record = self._build_record(class_name, real_row)
        self._document.append_paste(
            data_row,
            real_row,
            paste_image,
            paste_record=record,
        )
        self._refresh_total()
        self._image_canvas.set_canvas(preview_canvas)
        if self._on_disable_add is not None:
            self._on_disable_add()
        self._on_canvas_updated()
        return True

    def rename_selected(self) -> None:
        row = self.current_row()
        if row < 0:
            return
        names = list(self._get_object_names())
        text, ok = self._ask_label(self._parent, names)
        if not ok:
            return
        self.rename_row(row, text)

    def rename_row(self, row: int, new_name: str) -> None:
        if row < 0 or row >= self._document.total_pastes:
            return
        old_name = self._document.pimg_data[row][0]
        if old_name == new_name:
            return
        self._document.rename_paste(row, new_name)
        item = self._list_widget.item(row)
        if item is not None:
            item.setText(new_name)
        if new_name not in self._get_object_names():
            self._append_object_name(new_name)

    def delete_selected(self) -> None:
        self.remove_row(self.current_row())

    def remove_row(self, row: int) -> None:
        if row < 0 or row >= self._document.total_pastes:
            return
        self._document.remove_paste(row)
        self._list_widget.takeItem(row)
        self._refresh_total()
        if self._on_rows_changed is not None:
            self._on_rows_changed()

    def clear_all(self) -> None:
        if not self._confirm_clear_all(self._parent):
            return
        self._document.clear()
        self._list_widget.clear()
        self._refresh_total()
        if self._on_rows_changed is not None:
            self._on_rows_changed()

    def open_context_menu(self, pos) -> None:
        item = self._list_widget.itemAt(pos)
        if item is not None:
            self._list_widget.setCurrentItem(item)
        if self.current_row() < 0:
            return
        context = QtWidgets.QMenu(self._parent)
        action_rename = QAction("Rename", self._parent)
        action_delete = QAction("Delete", self._parent)
        context.addAction(action_rename)
        context.addAction(action_delete)
        action_rename.triggered.connect(self.rename_selected)
        action_delete.triggered.connect(self.delete_selected)
        context.exec(self._list_widget.viewport().mapToGlobal(pos))

    def _refresh_total(self) -> None:
        self._count_label.setText(f"Paste Images  (Total: {self._document.total_pastes})")
