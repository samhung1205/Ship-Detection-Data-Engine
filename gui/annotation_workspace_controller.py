"""
Workspace controller for GT selection, preview, and attribute-panel sync.
"""
from __future__ import annotations

from typing import Any, Callable


class AnnotationWorkspaceController:
    def __init__(
        self,
        *,
        document: Any,
        list_view: Any,
        attr_panel: Any,
        preview_controller: Any,
        on_delete_row: Callable[[int], None],
        on_rename_row: Callable[[int, str, str], None],
        on_clear_all: Callable[[], None],
    ) -> None:
        self._document = document
        self._list_view = list_view
        self._attr_panel = attr_panel
        self._preview_controller = preview_controller
        self._on_delete_row = on_delete_row
        self._on_rename_row = on_rename_row
        self._on_clear_all = on_clear_all

    def preview_current_row(self, *_args: object) -> None:
        row = self._list_view.current_index_row()
        if row >= 0:
            self._preview_controller.preview_row(row)

    def on_row_changed(self, row: int) -> None:
        if row < 0 or row >= len(self._document.box_attributes):
            self._preview_controller.clear_preview()
            self._attr_panel.set_enabled_editing(False)
            self._attr_panel.load_from_dict(self._document.attributes_or_default(-1))
            return
        self._preview_controller.preview_row(row)
        self._attr_panel.set_enabled_editing(True)
        self._attr_panel.load_from_dict(self._document.attributes_or_default(row))

    def on_attr_panel_changed(self) -> None:
        row = self._list_view.current_row()
        if row < 0 or row >= len(self._document.box_attributes):
            return
        self._document.set_box_attributes(row, self._attr_panel.to_dict())

    def on_recalc_size_tag(self) -> None:
        row = self._list_view.current_row()
        if row < 0 or row >= len(self._document.real_data) or row >= len(self._document.box_attributes):
            return
        size_tag = self._document.recalc_size_tag(row)
        self._attr_panel.combo_size.setCurrentText(size_tag)
        self._document.set_box_attributes(row, self._attr_panel.to_dict())

    def delete_row(self, row: int) -> None:
        self._preview_controller.clear_preview()
        self._on_delete_row(row)

    def rename_row(self, row: int, old_name: str, new_name: str) -> None:
        self._on_rename_row(row, old_name, new_name)

    def clear_all(self) -> None:
        self._preview_controller.clear_preview()
        self._on_clear_all()

    def clear_selection(self) -> None:
        self._list_view.set_current_row(-1)
        self.on_row_changed(-1)
