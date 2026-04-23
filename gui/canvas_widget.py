"""
Image canvas widget: scrollable view, zoom-scale redraw, bbox/paste overlay drawing.

Extracted from main_window for SDDE step 3 (CanvasWidget). Mouse handling is delegated
via hooks so MyWidget can switch between default / draw-rect / paste modes.
"""
from __future__ import annotations

from typing import Any, Callable, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

from .canvas_utils import (
    compute_magnifier_anchor,
    draw_boundary_boxes_on_canvas,
    draw_bboxes_on_canvas,
    draw_error_cases_overlay,
    draw_paste_images_on_canvas,
    draw_paste_zone_overlay,
    draw_predictions_on_canvas,
    draw_tile_grid_overview,
    draw_tile_overlay,
)

# Optional mouse event handler: receives the QMouseEvent from the image label.
MouseEventHandler = Optional[Callable]


class CanvasImageLabel(QtWidgets.QLabel):
    """QLabel that forwards mouse events to pluggable handlers."""

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._move_handler: MouseEventHandler = None
        self._press_handler: MouseEventHandler = None
        self._release_handler: MouseEventHandler = None
        self._leave_handler: MouseEventHandler = None

    def set_move_handler(self, fn: MouseEventHandler) -> None:
        self._move_handler = fn

    def set_press_handler(self, fn: MouseEventHandler) -> None:
        self._press_handler = fn

    def set_release_handler(self, fn: MouseEventHandler) -> None:
        self._release_handler = fn

    def set_leave_handler(self, fn: MouseEventHandler) -> None:
        self._leave_handler = fn

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._move_handler is not None:
            self._move_handler(event)
        else:
            super().mouseMoveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._press_handler is not None:
            self._press_handler(event)
        else:
            super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._release_handler is not None:
            self._release_handler(event)
        else:
            super().mouseReleaseEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._leave_handler is not None:
            self._leave_handler(event)
        else:
            super().leaveEvent(event)


class ImageCanvasWidget(QtWidgets.QWidget):
    """
    Scroll area + image label; holds the current display QPixmap (`canvas`).

    Use `redraw_scaled_overlay` to apply the same zoom + bbox + paste pipeline
    as the legacy MyWidget.set_img_ratio.
    """

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self._canvas: Optional[QPixmap] = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.scroll_area.setWidgetResizable(True)

        self.image_label = CanvasImageLabel()
        self.image_label.setAlignment(
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop
        )
        self._magnifier_label = QtWidgets.QLabel(self.image_label)
        self._magnifier_label.setFixedSize(160, 160)
        self._magnifier_label.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents, True
        )
        self._magnifier_label.setStyleSheet(
            "border: 2px solid #00BFFF; background: rgba(255, 255, 255, 235);"
        )
        self._magnifier_label.hide()
        self.scroll_area.setWidget(self.image_label)
        layout.addWidget(self.scroll_area)

    @property
    def canvas(self) -> Optional[QPixmap]:
        return self._canvas

    def set_canvas(self, pixmap: QPixmap, *, refresh: bool = True) -> None:
        """Set the working pixmap; optionally push it to the label."""
        self._canvas = pixmap
        if refresh and pixmap is not None:
            self.image_label.setPixmap(pixmap)

    def paint_label_only(self, pixmap: QPixmap) -> None:
        """Update the label without changing the stored working canvas (preview / highlight)."""

        self.image_label.setPixmap(pixmap)

    def sync_label_from_canvas(self) -> None:
        """Re-apply the working canvas to the label (e.g. after a temporary paint_label_only)."""

        if self._canvas is not None:
            self.image_label.setPixmap(self._canvas)

    def redraw_scaled_overlay(
        self,
        *,
        origin_canvas: QPixmap,
        ratio_value: int,
        origin_height: int,
        origin_width: int,
        hide_boxes: bool,
        bbox_data: list,
        pimg_data: list,
        paste_images: list[Any],
        paste_zone_rect: tuple[int, int, int, int] | None = None,
        predictions: list[Any] | None = None,
        show_predictions: bool = False,
        tile_rect: tuple[int, int, int, int] | None = None,
        tile_grid_rects: list[tuple[int, int, int, int]] | None = None,
        tile_grid_current_index: int = -1,
        boundary_rows: list[list] | None = None,
        boundary_labels: list[str] | None = None,
        error_cases: list[Any] | None = None,
        error_gt_boxes: list[tuple[str, float, float, float, float]] | None = None,
    ) -> tuple[float, int]:
        """
        Scale origin by zoom curve, draw boxes and paste images, update label.

        Parameters
        ----------
        tile_rect : (x, y, w, h) in origin-pixel space or None.
            When provided, a green tile boundary + dimming overlay is drawn.

        Returns (ratio_rate, qpixmap_height) for status display / reuse.
        """
        ratio_rate = pow(10, (ratio_value - 50) / 50)
        qpixmap_height = int(origin_height * ratio_rate)
        canvas = origin_canvas.scaledToHeight(
            qpixmap_height,
            Qt.TransformationMode.SmoothTransformation,
        )
        draw_paste_zone_overlay(
            canvas,
            paste_zone_rect,
            origin_width=origin_width,
            origin_height=origin_height,
        )
        if not hide_boxes:
            draw_bboxes_on_canvas(canvas, bbox_data + pimg_data)
        draw_paste_images_on_canvas(canvas, paste_images)
        if show_predictions and predictions:
            draw_predictions_on_canvas(
                canvas,
                predictions,
                origin_width=origin_width,
                origin_height=origin_height,
                show_confidence=True,
            )
        if tile_rect is not None:
            draw_tile_overlay(
                canvas,
                tile_rect[0], tile_rect[1], tile_rect[2], tile_rect[3],
                origin_width=origin_width,
                origin_height=origin_height,
            )
        elif tile_grid_rects:
            draw_tile_grid_overview(
                canvas,
                tile_grid_rects,
                current_index=tile_grid_current_index,
                origin_width=origin_width,
                origin_height=origin_height,
            )
        if boundary_rows:
            draw_boundary_boxes_on_canvas(
                canvas,
                boundary_rows,
                origin_width=origin_width,
                origin_height=origin_height,
                labels=boundary_labels,
            )
        if error_cases and error_gt_boxes is not None:
            draw_error_cases_overlay(
                canvas,
                error_cases,
                gt_boxes=error_gt_boxes,
                predictions=predictions or [],
                origin_width=origin_width,
                origin_height=origin_height,
            )
        self.set_canvas(canvas, refresh=True)
        return ratio_rate, qpixmap_height

    def set_mouse_move_handler(self, fn: MouseEventHandler) -> None:
        self.image_label.set_move_handler(fn)

    def set_mouse_press_handler(self, fn: MouseEventHandler) -> None:
        self.image_label.set_press_handler(fn)

    def set_mouse_release_handler(self, fn: MouseEventHandler) -> None:
        self.image_label.set_release_handler(fn)

    def set_mouse_leave_handler(self, fn: MouseEventHandler) -> None:
        self.image_label.set_leave_handler(fn)

    def show_magnifier(
        self,
        pixmap: QPixmap | None,
        *,
        cursor_x: int,
        cursor_y: int,
    ) -> None:
        if pixmap is None:
            self.hide_magnifier()
            return
        self._magnifier_label.setPixmap(pixmap)
        self._magnifier_label.resize(pixmap.size())
        x, y = compute_magnifier_anchor(
            cursor_x=cursor_x,
            cursor_y=cursor_y,
            image_width=self.image_label.width(),
            image_height=self.image_label.height(),
            preview_size=self._magnifier_label.width(),
        )
        self._magnifier_label.move(x, y)
        self._magnifier_label.show()
        self._magnifier_label.raise_()

    def hide_magnifier(self) -> None:
        self._magnifier_label.hide()
