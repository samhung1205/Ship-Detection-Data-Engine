"""
Annotation undo/redo controller for legacy box lists (data / real_data + listwidget).

Step 4 of SDDE refactor: command stack (max 50) without changing the on-disk row format yet.
"""
from __future__ import annotations

import copy
from typing import Any, List, Protocol, Tuple

from sdde.document import AnnotationBoxState


MAX_UNDO = 50

# Legacy row shapes (same as main_window):
# data:  [name, x1, y1, x2, y2, canvas_w, canvas_h]  (canvas coords may be origin-sized when loaded)
# real:  [name, rx1, ry1, rx2, ry2]


class AnnotationCommand(Protocol):
    def apply(self, w: Any) -> None: ...
    def unapply(self, w: Any) -> None: ...


def _refresh_canvas(w: Any) -> None:
    """Redraw scaled image + bbox / paste overlays (respects Hide Box)."""
    w.gt_list_view.set_total(len(w.real_data))
    w.set_img_ratio()


def _gt_document(w: Any):
    return w.gt_document


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
        doc = _gt_document(w)
        w.gt_list_view.add_item(self._label_text)
        if self._extended_object_list:
            w.object_list.append(self._label_text)
        doc.append_box(self._data_row, self._real_row)
        # Prefer full redraw over partial canvas hacks (matches set_img_ratio pipeline).
        if w.hideBox.isChecked():
            w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        doc.remove_box(-1)
        w.gt_list_view.remove_item(w.gt_list_view.count() - 1)
        if self._extended_object_list:
            w.object_list.pop()
        _refresh_canvas(w)


class RemoveBoxCommand:
    """Remove box at index (context menu delete)."""

    def __init__(self, index: int) -> None:
        self._index = index
        self._saved_state: AnnotationBoxState | None = None
        self._saved_label: str | None = None

    def apply(self, w: Any) -> None:
        doc = _gt_document(w)
        i = self._index
        self._saved_label = w.gt_list_view.item_text(i)
        self._saved_state = doc.remove_box(i)
        w.gt_list_view.remove_item(i)
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        assert self._saved_state is not None
        assert self._saved_label is not None
        i = self._index
        doc.insert_box(i, self._saved_state)
        w.gt_list_view.insert_item(i, self._saved_label)
        _refresh_canvas(w)


class ClearAllBoxesCommand:
    """Delete all box annotations (not paste images)."""

    def __init__(self) -> None:
        self._snapshot = None
        self._labels: List[str] = []

    def apply(self, w: Any) -> None:
        doc = _gt_document(w)
        self._snapshot = doc.snapshot()
        self._labels = [w.gt_list_view.item_text(i) for i in range(w.gt_list_view.count())]
        doc.clear()
        w.gt_list_view.clear()
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        assert self._snapshot is not None
        doc.restore(self._snapshot)
        w.gt_list_view.clear()
        for t in self._labels:
            w.gt_list_view.add_item(t)
        _refresh_canvas(w)


class RenameBoxCommand:
    """Rename class label for one box (list + data rows)."""

    def __init__(self, index: int, old_name: str, new_name: str) -> None:
        self._index = index
        self._old_name = old_name
        self._new_name = new_name
        self._extended_object_list = False

    def apply(self, w: Any) -> None:
        doc = _gt_document(w)
        i = self._index
        doc.rename_box(i, self._new_name)
        w.gt_list_view.rename_item(i, self._new_name)
        self._extended_object_list = False
        if self._new_name not in w.object_list:
            w.object_list.append(self._new_name)
            self._extended_object_list = True
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        i = self._index
        doc.rename_box(i, self._old_name)
        w.gt_list_view.rename_item(i, self._old_name)
        if self._extended_object_list:
            w.object_list.remove(self._new_name)
        _refresh_canvas(w)


class UpdateBoxGeometryCommand:
    """Replace one GT box geometry while preserving label + attributes."""

    def __init__(self, index: int, data_row: list, real_row: list) -> None:
        self._index = index
        self._next_state = AnnotationBoxState(
            data_row=copy.deepcopy(data_row),
            real_row=copy.deepcopy(real_row),
            attributes={},
        )
        self._previous_state: AnnotationBoxState | None = None

    def apply(self, w: Any) -> None:
        doc = _gt_document(w)
        if self._previous_state is None:
            self._previous_state = doc.box_state(self._index)
            self._next_state = AnnotationBoxState(
                data_row=copy.deepcopy(self._next_state.data_row),
                real_row=copy.deepcopy(self._next_state.real_row),
                attributes=copy.deepcopy(self._previous_state.attributes),
            )
        doc.replace_box(self._index, self._next_state)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        assert self._previous_state is not None
        doc.replace_box(self._index, self._previous_state)
        _refresh_canvas(w)


class BulkAppendBoxesCommand:
    """Append many boxes in one step (YOLO load). Undo drops all appended rows."""

    def __init__(self, blocks: List[Tuple[list, list, str]]) -> None:
        # Each block: (data_row, real_row, list_label)
        self._blocks = [(copy.deepcopy(d), copy.deepcopy(r), n) for d, r, n in blocks]

    def apply(self, w: Any) -> None:
        doc = _gt_document(w)
        for d_row, r_row, name in self._blocks:
            w.gt_list_view.add_item(name)
        doc.append_boxes(self._blocks)
        w.hideBox.setChecked(False)
        _refresh_canvas(w)

    def unapply(self, w: Any) -> None:
        doc = _gt_document(w)
        n = len(self._blocks)
        for _ in range(n):
            doc.remove_box(-1)
            w.gt_list_view.remove_item(w.gt_list_view.count() - 1)
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

    def add_box(
        self,
        data_row: list,
        real_row: list,
        label_text: str,
        *,
        extended_object_list: bool,
    ) -> None:
        self.apply(
            AddBoxCommand(
                data_row,
                real_row,
                label_text,
                extended_object_list=extended_object_list,
            )
        )

    def remove_box(self, index: int) -> None:
        self.apply(RemoveBoxCommand(index))

    def clear_all_boxes(self) -> None:
        self.apply(ClearAllBoxesCommand())

    def rename_box(self, index: int, old_name: str, new_name: str) -> None:
        self.apply(RenameBoxCommand(index, old_name, new_name))

    def update_box_geometry(
        self,
        index: int,
        data_row: list,
        real_row: list,
    ) -> None:
        self.apply(UpdateBoxGeometryCommand(index, data_row, real_row))

    def append_blocks(self, blocks: List[Tuple[list, list, str]]) -> None:
        self.apply(BulkAppendBoxesCommand(blocks))

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
