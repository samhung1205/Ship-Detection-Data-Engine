"""
Controller for temporary paste-row highlight previews on the image canvas.
"""
from __future__ import annotations

from typing import Any, Callable, Sequence

from PyQt6.QtGui import QColor, QPainter


class PastePreviewController:
    def __init__(
        self,
        *,
        document: Any,
        get_canvas: Callable[[], Any],
        image_canvas: Any,
        on_canvas_updated: Callable[[], None] | None = None,
    ) -> None:
        self._document = document
        self._get_canvas = get_canvas
        self._image_canvas = image_canvas
        self._on_canvas_updated = on_canvas_updated

    def preview_row(self, row: int) -> None:
        canvas = self._get_canvas()
        if canvas is None or row < 0 or row >= len(self._document.pimg_data):
            self.clear_preview()
            return

        row_data = self._document.pimg_data[row]
        if len(row_data) < 7:
            self.clear_preview()
            return

        x1, y1, x2, y2, width, height = row_data[1:]
        if width <= 0 or height <= 0:
            self.clear_preview()
            return

        scale_x = canvas.width() / width
        scale_y = canvas.height() / height
        x1 *= scale_x
        y1 *= scale_y
        x2 *= scale_x
        y2 *= scale_y

        copy_canvas = canvas.copy()
        qpainter = QPainter()
        qpainter.begin(copy_canvas)
        qpainter.fillRect(
            int(x1),
            int(y1),
            abs(int(x2 - x1)) + 1,
            abs(int(y2 - y1)) + 1,
            QColor(30, 144, 255, 120),
        )
        qpainter.end()
        self._image_canvas.paint_label_only(copy_canvas)
        self._notify_canvas_updated()

    def clear_preview(self) -> None:
        self._image_canvas.sync_label_from_canvas()
        self._notify_canvas_updated()

    def _notify_canvas_updated(self) -> None:
        if self._on_canvas_updated is not None:
            self._on_canvas_updated()
