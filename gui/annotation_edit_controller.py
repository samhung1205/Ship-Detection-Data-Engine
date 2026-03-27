"""
Controller for drag-move / corner-resize editing of committed GT boxes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PyQt6.QtGui import QColor

from .canvas_utils import draw_selection_overlay


HANDLE_RADIUS = 8
MIN_BOX_SIDE_PX = 2


@dataclass
class _DragSession:
    row: int
    mode: str
    left: float
    top: float
    right: float
    bottom: float
    offset_x: float = 0.0
    offset_y: float = 0.0


class AnnotationEditController:
    def __init__(
        self,
        *,
        document: Any,
        list_view: Any,
        get_canvas: Callable[[], Any],
        image_canvas: Any,
        render_canvas: Callable[[int | None], None],
        on_request_update_box: Callable[[int, int, int, int, int], bool],
        on_restore_preview: Callable[[int], None],
        on_canvas_updated: Callable[[], None] | None = None,
    ) -> None:
        self._document = document
        self._list_view = list_view
        self._get_canvas = get_canvas
        self._image_canvas = image_canvas
        self._render_canvas = render_canvas
        self._on_request_update_box = on_request_update_box
        self._on_restore_preview = on_restore_preview
        self._on_canvas_updated = on_canvas_updated
        self._drag: _DragSession | None = None

    def handle_press(self, event) -> bool:
        row = self._list_view.current_row()
        rect = self._row_canvas_rect(row)
        if rect is None:
            return False

        mx, my = self._event_point(event)
        hit = self._hit_target(rect, mx, my)
        if hit is None:
            return False

        left, top, right, bottom = rect
        if hit == "move":
            self._drag = _DragSession(
                row=row,
                mode=hit,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
                offset_x=mx - left,
                offset_y=my - top,
            )
        else:
            self._drag = _DragSession(
                row=row,
                mode=hit,
                left=left,
                top=top,
                right=right,
                bottom=bottom,
            )
        self._render_drag_preview(self._rect_for_pointer(self._drag, mx, my), row=row)
        return True

    def handle_move(self, event) -> bool:
        if self._drag is None:
            return False
        mx, my = self._event_point(event, clamp=True)
        self._render_drag_preview(self._rect_for_pointer(self._drag, mx, my), row=self._drag.row)
        return True

    def handle_release(self, event) -> bool:
        if self._drag is None:
            return False

        drag = self._drag
        self._drag = None
        mx, my = self._event_point(event, clamp=True)
        left, top, right, bottom = self._normalise_rect(self._rect_for_pointer(drag, mx, my))
        if right - left < MIN_BOX_SIDE_PX or bottom - top < MIN_BOX_SIDE_PX:
            self._restore_preview(drag.row)
            return False

        ok = self._on_request_update_box(
            drag.row,
            int(round(left)),
            int(round(top)),
            int(round(right)),
            int(round(bottom)),
        )
        if not ok:
            self._restore_preview(drag.row)
            return False

        self._on_restore_preview(drag.row)
        return True

    def cancel_active_drag(self) -> None:
        if self._drag is None:
            return
        row = self._drag.row
        self._drag = None
        self._restore_preview(row)

    def _row_canvas_rect(self, row: int) -> tuple[float, float, float, float] | None:
        canvas = self._get_canvas()
        if canvas is None or row < 0 or row >= len(self._document.data):
            return None
        row_data = self._document.data[row]
        if len(row_data) < 7:
            return None
        x1, y1, x2, y2, width, height = row_data[1:]
        if width <= 0 or height <= 0:
            return None
        scale_x = canvas.width() / width
        scale_y = canvas.height() / height
        return self._normalise_rect(
            (
                float(x1) * scale_x,
                float(y1) * scale_y,
                float(x2) * scale_x,
                float(y2) * scale_y,
            )
        )

    def _hit_target(
        self,
        rect: tuple[float, float, float, float],
        mx: float,
        my: float,
    ) -> str | None:
        left, top, right, bottom = rect
        for name, cx, cy in (
            ("tl", left, top),
            ("tr", right, top),
            ("bl", left, bottom),
            ("br", right, bottom),
        ):
            if abs(mx - cx) <= HANDLE_RADIUS and abs(my - cy) <= HANDLE_RADIUS:
                return name
        if left <= mx <= right and top <= my <= bottom:
            return "move"
        return None

    def _rect_for_pointer(
        self,
        drag: _DragSession,
        mx: float,
        my: float,
    ) -> tuple[float, float, float, float]:
        canvas = self._get_canvas()
        if canvas is None:
            return drag.left, drag.top, drag.right, drag.bottom

        if drag.mode == "move":
            width = drag.right - drag.left
            height = drag.bottom - drag.top
            max_left = max(0.0, float(canvas.width()) - width)
            max_top = max(0.0, float(canvas.height()) - height)
            left = min(max(mx - drag.offset_x, 0.0), max_left)
            top = min(max(my - drag.offset_y, 0.0), max_top)
            return left, top, left + width, top + height

        if drag.mode == "tl":
            return mx, my, drag.right, drag.bottom
        if drag.mode == "tr":
            return drag.left, my, mx, drag.bottom
        if drag.mode == "bl":
            return mx, drag.top, drag.right, my
        return drag.left, drag.top, mx, my

    def _render_drag_preview(
        self,
        rect: tuple[float, float, float, float],
        *,
        row: int,
    ) -> None:
        self._render_canvas(row)
        canvas = self._get_canvas()
        if canvas is None:
            return
        left, top, right, bottom = self._normalise_rect(rect)
        draw_selection_overlay(
            canvas,
            x1=int(round(left)),
            y1=int(round(top)),
            x2=int(round(right)),
            y2=int(round(bottom)),
            fill_color=QColor(30, 144, 255, 120),
            outline_color=QColor(30, 144, 255),
        )
        self._image_canvas.sync_label_from_canvas()
        self._notify_canvas_updated()

    def _restore_preview(self, row: int) -> None:
        self._render_canvas(None)
        self._on_restore_preview(row)

    def _event_point(self, event, *, clamp: bool = False) -> tuple[float, float]:
        canvas = self._get_canvas()
        x = float(event.position().x())
        y = float(event.position().y())
        if not clamp or canvas is None:
            return x, y
        max_x = max(0.0, float(canvas.width()))
        max_y = max(0.0, float(canvas.height()))
        return min(max(x, 0.0), max_x), min(max(y, 0.0), max_y)

    @staticmethod
    def _normalise_rect(
        rect: tuple[float, float, float, float],
    ) -> tuple[float, float, float, float]:
        x1, y1, x2, y2 = rect
        return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)

    def _notify_canvas_updated(self) -> None:
        if self._on_canvas_updated is not None:
            self._on_canvas_updated()
