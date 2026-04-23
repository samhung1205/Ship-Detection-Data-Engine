"""
Controller for GT action entry points.

This keeps prompt-driven add-box flows and high-level command routing out of
``main_window.py`` while reusing the existing undo/redo stack.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from PyQt6 import QtWidgets


AskLabelFn = Callable[[Any, Sequence[str]], tuple[str, bool]]


def _default_ask_label(parent: Any, names: Sequence[str]) -> tuple[str, bool]:
    return QtWidgets.QInputDialog().getItem(
        parent,
        "",
        "Enter object name",
        list(names),
        0,
        False,
    )


class AnnotationActionsController:
    def __init__(
        self,
        *,
        parent: Any,
        command_controller: Any,
        list_view: Any,
        get_object_names: Callable[[], Sequence[str]],
        get_canvas: Callable[[], Any],
        get_origin_size: Callable[[], tuple[int, int]],
        on_add_cancelled: Callable[[], None] | None = None,
        ask_label: AskLabelFn | None = None,
    ) -> None:
        self._parent = parent
        self._command_controller = command_controller
        self._list_view = list_view
        self._get_object_names = get_object_names
        self._get_canvas = get_canvas
        self._get_origin_size = get_origin_size
        self._on_add_cancelled = on_add_cancelled
        self._ask_label = ask_label or _default_ask_label

    def prompt_add_box(self, x1: int, y1: int, x2: int, y2: int) -> bool:
        item, ok = self._ask_label(self._parent, list(self._get_object_names()))
        if not ok:
            if self._on_add_cancelled is not None:
                self._on_add_cancelled()
            return False
        payload = self.build_add_box_from_rect(
            item=item,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if payload is None:
            return False
        self.add_box(*payload, select_new_row=True)
        return True

    def build_box_rows_from_rect(
        self,
        *,
        item: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[list, list] | None:
        canvas = self._get_canvas()
        origin_width, origin_height = self._get_origin_size()
        if canvas is None or origin_width <= 0 or origin_height <= 0:
            return None
        canvas_width = canvas.width()
        canvas_height = canvas.height()
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        real_x1 = int(x1 * origin_width / canvas_width)
        real_y1 = int(y1 * origin_height / canvas_height)
        real_x2 = int(x2 * origin_width / canvas_width)
        real_y2 = int(y2 * origin_height / canvas_height)
        data_row = [item, x1, y1, x2, y2, canvas_width, canvas_height]
        real_row = [item, real_x1, real_y1, real_x2, real_y2]
        return data_row, real_row

    def build_add_box_from_rect(
        self,
        *,
        item: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[list, list, str, bool] | None:
        rows = self.build_box_rows_from_rect(
            item=item,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if rows is None:
            return None
        data_row, real_row = rows
        return data_row, real_row, item, item not in self._get_object_names()

    def build_add_box_from_prediction(
        self,
        pred: Any,
    ) -> tuple[list, list, str, bool] | None:
        canvas = self._get_canvas()
        origin_width, origin_height = self._get_origin_size()
        if canvas is None or origin_width <= 0 or origin_height <= 0:
            return None
        canvas_width = canvas.width()
        canvas_height = canvas.height()
        if canvas_width <= 0 or canvas_height <= 0:
            return None
        item = pred.class_name
        data_row = [
            item,
            pred.x1 * canvas_width / origin_width,
            pred.y1 * canvas_height / origin_height,
            pred.x2 * canvas_width / origin_width,
            pred.y2 * canvas_height / origin_height,
            canvas_width,
            canvas_height,
        ]
        real_row = [item, int(pred.x1), int(pred.y1), int(pred.x2), int(pred.y2)]
        return data_row, real_row, item, item not in self._get_object_names()

    def add_box(
        self,
        data_row: list,
        real_row: list,
        label_text: str,
        extended_object_list: bool,
        *,
        select_new_row: bool = False,
    ) -> None:
        self._command_controller.add_box(
            data_row,
            real_row,
            label_text,
            extended_object_list=extended_object_list,
        )
        if select_new_row and self._list_view.count() > 0:
            self._list_view.set_current_row(self._list_view.count() - 1)

    def append_blocks(self, blocks: list[tuple[list, list, str]]) -> None:
        self._command_controller.append_blocks(blocks)

    def replace_with_blocks(self, blocks: list[tuple[list, list, str]]) -> None:
        self._command_controller.replace_all_boxes(blocks)

    def remove_row(self, row: int) -> None:
        self._command_controller.remove_box(row)

    def rename_row(self, row: int, old_name: str, new_name: str) -> None:
        self._command_controller.rename_box(row, old_name, new_name)

    def update_box_geometry(self, row: int, data_row: list, real_row: list) -> None:
        self._command_controller.update_box_geometry(row, data_row, real_row)

    def update_box_from_rect(
        self,
        *,
        row: int,
        item: str,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
        select_row: bool = False,
    ) -> bool:
        rows = self.build_box_rows_from_rect(
            item=item,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
        )
        if rows is None:
            return False
        data_row, real_row = rows
        self.update_box_geometry(row, data_row, real_row)
        if select_row and 0 <= row < self._list_view.count():
            self._list_view.set_current_row(row)
        return True

    def clear_all(self) -> None:
        self._command_controller.clear_all_boxes()
