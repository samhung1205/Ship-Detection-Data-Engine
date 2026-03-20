"""
Annotation undo/redo controller for legacy box lists (data / real_data + listwidget).

Step 4 of SDDE refactor: command stack (max 50) without changing the on-disk row format yet.
"""
from __future__ import annotations

import copy
from typing import Any, List, Protocol, Tuple


MAX_UNDO = 50

# Legacy row shapes (same as main_window):
# data:  [name, x1, y1, x2, y2, canvas_w, canvas_h]  (canvas coords may be origin-sized when loaded)
# real:  [name, rx1, ry1, rx2, ry2]


class AnnotationCommand(Protocol):
    def apply(self, w: Any) -> None: ...
    def unapply(self, w: Any) -> None: ...


def _refresh_canvas(w: Any) -> None:
    """Redraw scaled image + bbox / paste overlays (respects Hide Box)."""
    w.label_list.setText(f'Box Labels  (Total: {len(w.real_data)})')
    w.set_img_ratio()


def _append_box_attribute_row(w: Any) -> None:
    if hasattr(w, "append_box_attributes_row"):
        w.append_box_attributes_row()


def _pop_last_box_attribute_row(w: Any) -> None:
    if hasattr(w, "box_attributes") and w.box_attributes:
        w.box_attributes.pop()


class AddBoxCommand:
    """Append one manual box (after user confirms class in qInput)."""

    def __init__(
        self,
        data_row: list,
        real_row: list,
        label_text: str,
        *,
        extended_object_list: bool,
    ) -> None:
        self._data_row = copy.deepcopy(data_row)
        self._real_row = copy.deepcopy(real_row)
        self._label_text = label_text
        self._extended_object_list = extended_object_list

    def apply(self, w: Any) -> None:
        w.listwidget.addItem(self._label_text)
        if self._extended_object_list:
            w.object_list.append(self._label_text)
        w.data.append(copy.deepcopy(self._data_row))
        w.real_data.append(copy.deepcopy(self._real_row))
        # Prefer full redraw over partial canvas hacks (matches set_img_ratio pipeline).
        if w.hideBox.isChecked():
            w.hideBox.setChecked(False)
        _append_box_attribute_row(w)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        _pop_last_box_attribute_row(w)
        w.data.pop()
        w.real_data.pop()
        w.listwidget.takeItem(w.listwidget.count() - 1)
        if self._extended_object_list:
            w.object_list.pop()
        _refresh_canvas(w)


class RemoveBoxCommand:
    """Remove box at index (context menu delete)."""

    def __init__(self, index: int) -> None:
        self._index = index
        self._saved_data: list | None = None
        self._saved_real: list | None = None
        self._saved_label: str | None = None
        self._saved_attr: dict | None = None

    def apply(self, w: Any) -> None:
        i = self._index
        self._saved_data = copy.deepcopy(w.data[i])
        self._saved_real = copy.deepcopy(w.real_data[i])
        self._saved_label = w.listwidget.item(i).text()
        if hasattr(w, "box_attributes") and i < len(w.box_attributes):
            self._saved_attr = copy.deepcopy(w.box_attributes.pop(i))
        else:
            self._saved_attr = None
        w.data.pop(i)
        w.real_data.pop(i)
        w.listwidget.takeItem(i)
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        assert self._saved_data is not None and self._saved_real is not None
        assert self._saved_label is not None
        i = self._index
        w.data.insert(i, copy.deepcopy(self._saved_data))
        w.real_data.insert(i, copy.deepcopy(self._saved_real))
        w.listwidget.insertItem(i, self._saved_label)
        if self._saved_attr is not None and hasattr(w, "box_attributes"):
            w.box_attributes.insert(i, copy.deepcopy(self._saved_attr))
        _refresh_canvas(w)


class ClearAllBoxesCommand:
    """Delete all box annotations (not paste images)."""

    def __init__(self) -> None:
        self._snap_data: List[list] = []
        self._snap_real: List[list] = []
        self._labels: List[str] = []
        self._snap_attrs: List[dict] = []

    def apply(self, w: Any) -> None:
        self._snap_data = copy.deepcopy(w.data)
        self._snap_real = copy.deepcopy(w.real_data)
        self._labels = [w.listwidget.item(i).text() for i in range(w.listwidget.count())]
        if hasattr(w, "box_attributes"):
            self._snap_attrs = copy.deepcopy(w.box_attributes)
            w.box_attributes.clear()
        else:
            self._snap_attrs = []
        w.data.clear()
        w.real_data.clear()
        w.listwidget.clear()
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        w.data[:] = copy.deepcopy(self._snap_data)
        w.real_data[:] = copy.deepcopy(self._snap_real)
        w.listwidget.clear()
        for t in self._labels:
            w.listwidget.addItem(t)
        if hasattr(w, "box_attributes"):
            w.box_attributes[:] = copy.deepcopy(self._snap_attrs)
        _refresh_canvas(w)


class RenameBoxCommand:
    """Rename class label for one box (list + data rows)."""

    def __init__(self, index: int, old_name: str, new_name: str) -> None:
        self._index = index
        self._old_name = old_name
        self._new_name = new_name
        self._extended_object_list = False

    def apply(self, w: Any) -> None:
        i = self._index
        w.data[i][0] = self._new_name
        w.real_data[i][0] = self._new_name
        w.listwidget.item(i).setText(self._new_name)
        self._extended_object_list = False
        if self._new_name not in w.object_list:
            w.object_list.append(self._new_name)
            self._extended_object_list = True
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        i = self._index
        w.data[i][0] = self._old_name
        w.real_data[i][0] = self._old_name
        w.listwidget.item(i).setText(self._old_name)
        if self._extended_object_list:
            w.object_list.remove(self._new_name)
        _refresh_canvas(w)


class BulkAppendBoxesCommand:
    """Append many boxes in one step (YOLO load). Undo drops all appended rows."""

    def __init__(self, blocks: List[Tuple[list, list, str]]) -> None:
        # Each block: (data_row, real_row, list_label)
        self._blocks = [(copy.deepcopy(d), copy.deepcopy(r), n) for d, r, n in blocks]

    def apply(self, w: Any) -> None:
        for d_row, r_row, name in self._blocks:
            w.data.append(copy.deepcopy(d_row))
            w.real_data.append(copy.deepcopy(r_row))
            w.listwidget.addItem(name)
            _append_box_attribute_row(w)
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        n = len(self._blocks)
        for _ in range(n):
            _pop_last_box_attribute_row(w)
            w.data.pop()
            w.real_data.pop()
            w.listwidget.takeItem(w.listwidget.count() - 1)
        _refresh_canvas(w)


class AnnotationController:
    def __init__(self, widget: Any) -> None:
        self._w = widget
        self._undo: List[AnnotationCommand] = []
        self._redo: List[AnnotationCommand] = []

    def reset(self) -> None:
        self._undo.clear()
        self._redo.clear()

    def apply(self, cmd: AnnotationCommand) -> None:
        cmd.apply(self._w)
        self._undo.append(cmd)
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)
        self._redo.clear()

    def undo(self) -> None:
        if not self._undo:
            return
        cmd = self._undo.pop()
        cmd.unapply(self._w)
        self._redo.append(cmd)

    def redo(self) -> None:
        if not self._redo:
            return
        cmd = self._redo.pop()
        cmd.apply(self._w)
        self._undo.append(cmd)
        if len(self._undo) > MAX_UNDO:
            self._undo.pop(0)

    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)
