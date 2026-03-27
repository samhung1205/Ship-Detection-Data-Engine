"""
Thin UI adapter for the GT annotation list projection.

This keeps the legacy Qt widgets in place while centralising list/count updates
outside ``main_window.py``.
"""
from __future__ import annotations

from typing import Any, Sequence


class AnnotationListView:
    def __init__(
        self,
        *,
        count_label: Any,
        list_widget: Any,
        title: str = "Box Labels",
    ) -> None:
        self._count_label = count_label
        self._list_widget = list_widget
        self._title = title

    def set_total(self, total: int) -> None:
        self._count_label.setText(f"{self._title}  (Total: {total})")

    def sync_labels(self, labels: Sequence[str]) -> None:
        self.clear()
        for label in labels:
            self.add_item(label)
        self.set_total(len(labels))

    def add_item(self, text: str) -> None:
        self._list_widget.addItem(text)

    def insert_item(self, index: int, text: str) -> None:
        self._list_widget.insertItem(index, text)

    def remove_item(self, index: int) -> None:
        self._list_widget.takeItem(index)

    def rename_item(self, index: int, text: str) -> None:
        self._list_widget.item(index).setText(text)

    def item_text(self, index: int) -> str:
        return self._list_widget.item(index).text()

    def clear(self) -> None:
        self._list_widget.clear()

    def count(self) -> int:
        return self._list_widget.count()

    def set_current_row(self, row: int) -> None:
        self._list_widget.setCurrentRow(row)

    def current_row(self) -> int:
        return self._list_widget.currentRow()

    def current_index_row(self) -> int:
        return self._list_widget.currentIndex().row()

    def viewport_global_pos(self, pos):
        return self._list_widget.viewport().mapToGlobal(pos)
