"""
Controller for manual GT box drawing interactions on the image canvas.
"""
from __future__ import annotations

from typing import Any, Callable

from PyQt6.QtGui import QColor, QPainter, QPen


class AnnotationDrawController:
    def __init__(
        self,
        *,
        get_canvas: Callable[[], Any],
        image_canvas: Any,
        on_prepare_draw_mode: Callable[[], None],
        on_clicked_position: Callable[[int, int], None],
        on_request_add_box: Callable[[int, int, int, int], bool],
        on_reset_view: Callable[[], None],
        on_canvas_updated: Callable[[], None] | None = None,
    ) -> None:
        self._get_canvas = get_canvas
        self._image_canvas = image_canvas
        self._on_prepare_draw_mode = on_prepare_draw_mode
        self._on_clicked_position = on_clicked_position
        self._on_request_add_box = on_request_add_box
        self._on_reset_view = on_reset_view
        self._on_canvas_updated = on_canvas_updated
        self._start: tuple[int, int] | None = None

    def enter_draw_mode(self) -> None:
        self.clear_pending()
        self._on_prepare_draw_mode()
        self._image_canvas.set_mouse_press_handler(self.handle_press)

    def clear_pending(self) -> None:
        self._start = None

    def handle_press(self, event) -> None:
        mx = int(event.position().x())
        my = int(event.position().y())
        self._on_clicked_position(mx, my)
        canvas = self._get_canvas()
        if canvas is None:
            return
        if mx >= canvas.width() or my >= canvas.height():
            return

        self._draw_point(canvas, mx, my)
        if self._start is None:
            self._start = (mx, my)
            return

        x1, y1 = self._start
        self.clear_pending()
        if mx > x1 and my > y1:
            self._draw_rect(canvas, x1, y1, mx, my)
            self._on_request_add_box(x1, y1, mx, my)
            return
        self._on_reset_view()

    def _draw_point(self, canvas: Any, x: int, y: int) -> None:
        qpainter = QPainter()
        qpainter.begin(canvas)
        qpainter.setPen(QPen(QColor("#00ff00"), 3))
        qpainter.drawPoint(x, y)
        qpainter.end()
        self._image_canvas.sync_label_from_canvas()
        self._notify_canvas_updated()

    def _draw_rect(self, canvas: Any, x1: int, y1: int, x2: int, y2: int) -> None:
        qpainter = QPainter()
        qpainter.begin(canvas)
        qpainter.setPen(QPen(QColor("#00ff00"), 1))
        qpainter.drawRect(x1, y1, abs(x1 - x2), abs(y1 - y2))
        qpainter.end()
        self._image_canvas.sync_label_from_canvas()
        self._notify_canvas_updated()

    def _notify_canvas_updated(self) -> None:
        if self._on_canvas_updated is not None:
            self._on_canvas_updated()
