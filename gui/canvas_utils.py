"""
Canvas drawing utilities for bounding boxes and paste images.
"""
from collections.abc import Sequence
from typing import Any

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap, QFont


def draw_bboxes_on_canvas(canvas: QPixmap, bbox_datas: list) -> None:
    """Draw bounding boxes from bbox_datas onto canvas."""
    qpainter = QPainter()
    qpainter.begin(canvas)
    for datas in bbox_datas:
        x1, y1, x2, y2, w, h = datas[1:]
        x1 *= canvas.width() / w
        y1 *= canvas.height() / h
        x2 *= canvas.width() / w
        y2 *= canvas.height() / h
        qpainter.setPen(QPen(QColor('#00ff00'), 3))
        qpainter.drawPoint(int(x1), int(y1))
        qpainter.setPen(QPen(QColor('#00ff00'), 3))
        qpainter.drawPoint(int(x2), int(y2))
        qpainter.setPen(QPen(QColor('#00ff00'), 1))
        qpainter.drawRect(int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1)))
    qpainter.end()


def draw_selection_overlay(
    canvas: QPixmap,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    fill_color: QColor | None = None,
    outline_color: QColor | None = None,
    handle_size: int = 6,
) -> None:
    """Draw a highlighted selection box with corner handles."""
    left = min(int(x1), int(x2))
    top = min(int(y1), int(y2))
    right = max(int(x1), int(x2))
    bottom = max(int(y1), int(y2))
    width = max(0, right - left)
    height = max(0, bottom - top)
    outline = outline_color or QColor(30, 144, 255)

    qpainter = QPainter()
    qpainter.begin(canvas)
    if fill_color is not None:
        qpainter.fillRect(left, top, width + 1, height + 1, fill_color)
    qpainter.setPen(QPen(outline, 1))
    qpainter.drawRect(left, top, width, height)
    half = max(1, handle_size // 2)
    for px, py in ((left, top), (right, top), (left, bottom), (right, bottom)):
        qpainter.fillRect(px - half, py - half, handle_size, handle_size, outline)
    qpainter.end()


def _select_paste_overlay_payload(
    payload: Sequence[Any],
    *,
    prefer_export_geometry: bool = False,
) -> tuple[Any, float, float, float, float]:
    if prefer_export_geometry and len(payload) >= 10:
        return (
            payload[5],
            float(payload[6]),
            float(payload[7]),
            float(payload[8]),
            float(payload[9]),
        )
    if len(payload) < 5:
        raise ValueError("paste image payload must contain at least 5 values")
    return (
        payload[0],
        float(payload[1]),
        float(payload[2]),
        float(payload[3]),
        float(payload[4]),
    )


def draw_paste_images_on_canvas(
    canvas: QPixmap,
    paste_images: list,
    *,
    prefer_export_geometry: bool = False,
) -> None:
    """Draw paste images onto canvas."""
    from PyQt6.QtCore import QRect

    qpainter = QPainter()
    qpainter.begin(canvas)
    qpainter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    for data in paste_images:
        img, norm_x, norm_y, norm_w, norm_h = _select_paste_overlay_payload(
            data,
            prefer_export_geometry=prefer_export_geometry,
        )
        x = norm_x * canvas.width()
        y = norm_y * canvas.height()
        w = norm_w * canvas.width()
        h = norm_h * canvas.height()
        qpainter.drawImage(QRect(int(x), int(y), int(w), int(h)), img)
    qpainter.end()


def draw_paste_zone_overlay(
    canvas: QPixmap,
    zone_rect: tuple[int, int, int, int] | None,
    *,
    origin_width: int,
    origin_height: int,
    label: str = "Smart zone",
) -> None:
    if zone_rect is None or origin_width <= 0 or origin_height <= 0:
        return

    x1, y1, x2, y2 = zone_rect
    left = int(round(x1 * canvas.width() / origin_width))
    top = int(round(y1 * canvas.height() / origin_height))
    right = int(round(x2 * canvas.width() / origin_width))
    bottom = int(round(y2 * canvas.height() / origin_height))
    width = max(0, right - left)
    height = max(0, bottom - top)

    qpainter = QPainter()
    qpainter.begin(canvas)
    color = QColor("#2D98DA")
    qpainter.fillRect(left, top, width + 1, height + 1, QColor(45, 152, 218, 24))
    qpainter.setPen(QPen(color, 2, Qt.PenStyle.DashLine))
    qpainter.drawRect(left, top, width, height)
    qpainter.fillRect(left, max(0, top - 13), 70, 14, QColor(45, 152, 218, 38))
    qpainter.setPen(QPen(color, 1))
    qpainter.drawText(left + 3, max(12, top - 2), label)
    qpainter.end()


def draw_predictions_on_canvas(
    canvas: QPixmap,
    predictions: list[Any],
    *,
    origin_width: int,
    origin_height: int,
    show_confidence: bool = True,
) -> None:
    """
    Draw prediction HBB in origin pixel space onto scaled canvas.
    Only draws items with pred_status == 'predicted' (or 'edited').
    """
    from PyQt6.QtCore import Qt as QtCoreQt

    if origin_width <= 0 or origin_height <= 0:
        return

    cw, ch = canvas.width(), canvas.height()
    qpainter = QPainter()
    qpainter.begin(canvas)
    pen = QPen(QColor("#FF6600"))
    pen.setWidth(2)
    pen.setStyle(QtCoreQt.PenStyle.DashLine)
    qpainter.setPen(pen)
    font = QFont()
    font.setPointSize(8)
    qpainter.setFont(font)

    for p in predictions:
        status = getattr(p, "pred_status", "")
        if status not in ("predicted", "edited"):
            continue
        x1 = float(p.x1) * cw / origin_width
        y1 = float(p.y1) * ch / origin_height
        x2 = float(p.x2) * cw / origin_width
        y2 = float(p.y2) * ch / origin_height
        qpainter.drawRect(int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1)))
        if show_confidence:
            label = f"{getattr(p, 'class_name', '')} {float(p.confidence):.2f}"
            qpainter.setPen(QPen(QColor("#FF3300")))
            qpainter.drawText(int(x1) + 2, max(12, int(y1) - 2), label)
            qpainter.setPen(pen)
    qpainter.end()


def draw_error_cases_overlay(
    canvas: QPixmap,
    error_cases: list[Any],
    *,
    gt_boxes: Sequence[tuple[str, float, float, float, float]],
    predictions: Sequence[Any],
    origin_width: int,
    origin_height: int,
) -> None:
    """
    Draw GT-vs-prediction pairing results on top of the existing GT/pred overlay.

    Matched cases are shown with colored center-to-center lines and an IoU label.
    FP / FN rows receive a small badge near the unmatched box.
    """
    if origin_width <= 0 or origin_height <= 0 or not error_cases:
        return

    try:
        from sdde.error_analysis import (
            ERROR_DUPLICATE,
            ERROR_FN,
            ERROR_FP,
            ERROR_LOCALIZATION,
            ERROR_TP,
            ERROR_WRONG_CLASS,
        )
    except ImportError:  # pragma: no cover - defensive fallback for local imports
        return

    color_map = {
        ERROR_TP: QColor("#00B894"),
        ERROR_WRONG_CLASS: QColor("#D63031"),
        ERROR_LOCALIZATION: QColor("#0984E3"),
        ERROR_DUPLICATE: QColor("#F39C12"),
        ERROR_FP: QColor("#C0392B"),
        ERROR_FN: QColor("#6C5CE7"),
    }

    cw, ch = canvas.width(), canvas.height()
    qpainter = QPainter()
    qpainter.begin(canvas)
    font = QFont()
    font.setPointSize(8)
    qpainter.setFont(font)

    for case in error_cases:
        color = color_map.get(getattr(case, "error_type", ""), QColor("#444444"))

        if case.gt_index is not None and case.pred_index is not None:
            if case.gt_index < 0 or case.gt_index >= len(gt_boxes):
                continue
            if case.pred_index < 0 or case.pred_index >= len(predictions):
                continue
            gt = gt_boxes[case.gt_index]
            pred = predictions[case.pred_index]
            gx1 = float(gt[1]) * cw / origin_width
            gy1 = float(gt[2]) * ch / origin_height
            gx2 = float(gt[3]) * cw / origin_width
            gy2 = float(gt[4]) * ch / origin_height
            px1 = float(pred.x1) * cw / origin_width
            py1 = float(pred.y1) * ch / origin_height
            px2 = float(pred.x2) * cw / origin_width
            py2 = float(pred.y2) * ch / origin_height
            gcx = int(round((gx1 + gx2) / 2.0))
            gcy = int(round((gy1 + gy2) / 2.0))
            pcx = int(round((px1 + px2) / 2.0))
            pcy = int(round((py1 + py2) / 2.0))
            qpainter.setPen(QPen(color, 2, Qt.PenStyle.DashDotLine))
            qpainter.drawLine(gcx, gcy, pcx, pcy)
            label = f"{case.error_type} {float(case.iou):.2f}"
            lx = int(round((gcx + pcx) / 2.0))
            ly = int(round((gcy + pcy) / 2.0))
            qpainter.fillRect(lx - 2, max(0, ly - 11), 68, 14, QColor(color.red(), color.green(), color.blue(), 40))
            qpainter.setPen(QPen(color, 1))
            qpainter.drawText(lx, ly, label)
            continue

        if case.error_type == ERROR_FP and case.pred_index is not None:
            if case.pred_index < 0 or case.pred_index >= len(predictions):
                continue
            pred = predictions[case.pred_index]
            x = int(float(pred.x1) * cw / origin_width)
            y = int(float(pred.y1) * ch / origin_height)
            qpainter.fillRect(x, max(0, y - 11), 20, 14, QColor(color.red(), color.green(), color.blue(), 40))
            qpainter.setPen(QPen(color, 1))
            qpainter.drawText(x + 2, max(12, y - 1), "FP")
            continue

        if case.error_type == ERROR_FN and case.gt_index is not None:
            if case.gt_index < 0 or case.gt_index >= len(gt_boxes):
                continue
            gt = gt_boxes[case.gt_index]
            x = int(float(gt[1]) * cw / origin_width)
            y = int(float(gt[2]) * ch / origin_height)
            qpainter.fillRect(x, max(0, y - 11), 20, 14, QColor(color.red(), color.green(), color.blue(), 40))
            qpainter.setPen(QPen(color, 1))
            qpainter.drawText(x + 2, max(12, y - 1), "FN")

    qpainter.end()


def draw_boundary_boxes_on_canvas(
    canvas: QPixmap,
    boundary_rows: list[list],
    *,
    origin_width: int,
    origin_height: int,
    labels: list[str] | None = None,
) -> None:
    """
    Highlight GT boxes that cross the current tile boundary.

    Rows are expected in origin-pixel space: [class_name, x1, y1, x2, y2].
    """
    from PyQt6.QtCore import Qt as QtCoreQt

    if origin_width <= 0 or origin_height <= 0 or not boundary_rows:
        return

    cw, ch = canvas.width(), canvas.height()
    qpainter = QPainter()
    qpainter.begin(canvas)

    pen = QPen(QColor("#FFB000"))
    pen.setWidth(2)
    pen.setStyle(QtCoreQt.PenStyle.DashDotLine)
    qpainter.setPen(pen)
    font = QFont()
    font.setPointSize(8)
    qpainter.setFont(font)

    for idx, row in enumerate(boundary_rows):
        x1 = float(row[1]) * cw / origin_width
        y1 = float(row[2]) * ch / origin_height
        x2 = float(row[3]) * cw / origin_width
        y2 = float(row[4]) * ch / origin_height
        left = min(int(x1), int(x2))
        top = min(int(y1), int(y2))
        width = abs(int(x2 - x1))
        height = abs(int(y2 - y1))
        qpainter.fillRect(left, top, width + 1, height + 1, QColor(255, 176, 0, 45))
        qpainter.drawRect(left, top, width, height)
        if labels and idx < len(labels):
            qpainter.drawText(left + 2, max(12, top - 2), labels[idx])
    qpainter.end()


def draw_tile_overlay(
    canvas: QPixmap,
    tile_x: int,
    tile_y: int,
    tile_w: int,
    tile_h: int,
    *,
    origin_width: int,
    origin_height: int,
) -> None:
    """
    Draw a tile boundary on the scaled canvas: dim the area outside the
    current tile and draw a green rect around it.
    """
    from PyQt6.QtCore import Qt as QtCoreQt

    if origin_width <= 0 or origin_height <= 0:
        return

    cw, ch = canvas.width(), canvas.height()
    sx = cw / origin_width
    sy = ch / origin_height
    rx = int(tile_x * sx)
    ry = int(tile_y * sy)
    rw = int(tile_w * sx)
    rh = int(tile_h * sy)

    qpainter = QPainter()
    qpainter.begin(canvas)

    dim = QColor(0, 0, 0, 100)
    if ry > 0:
        qpainter.fillRect(0, 0, cw, ry, dim)
    if ry + rh < ch:
        qpainter.fillRect(0, ry + rh, cw, ch - ry - rh, dim)
    if rx > 0:
        qpainter.fillRect(0, ry, rx, rh, dim)
    if rx + rw < cw:
        qpainter.fillRect(rx + rw, ry, cw - rx - rw, rh, dim)

    pen = QPen(QColor("#00CC66"))
    pen.setWidth(2)
    pen.setStyle(QtCoreQt.PenStyle.SolidLine)
    qpainter.setPen(pen)
    qpainter.drawRect(rx, ry, rw, rh)

    font = QFont()
    font.setPointSize(9)
    qpainter.setFont(font)
    qpainter.setPen(QPen(QColor("#00CC66")))
    qpainter.drawText(rx + 4, max(12, ry - 4), f"Tile ({tile_x},{tile_y}) {tile_w}x{tile_h}")
    qpainter.end()


def draw_tile_grid_overview(
    canvas: QPixmap,
    tile_rects: list[tuple[int, int, int, int]],
    *,
    current_index: int,
    origin_width: int,
    origin_height: int,
) -> None:
    """Draw the full tile grid on the scaled canvas and highlight the current tile."""
    from PyQt6.QtCore import Qt as QtCoreQt

    if origin_width <= 0 or origin_height <= 0 or not tile_rects:
        return

    cw, ch = canvas.width(), canvas.height()
    sx = cw / origin_width
    sy = ch / origin_height

    qpainter = QPainter()
    qpainter.begin(canvas)
    grid_pen = QPen(QColor(120, 128, 140, 120))
    grid_pen.setWidth(1)
    grid_pen.setStyle(QtCoreQt.PenStyle.SolidLine)
    highlight_pen = QPen(QColor("#00CC66"))
    highlight_pen.setWidth(2)
    highlight_pen.setStyle(QtCoreQt.PenStyle.SolidLine)
    highlight_fill = QColor(0, 204, 102, 36)

    font = QFont()
    font.setPointSize(9)
    qpainter.setFont(font)

    for idx, (tile_x, tile_y, tile_w, tile_h) in enumerate(tile_rects):
        rx = int(tile_x * sx)
        ry = int(tile_y * sy)
        rw = max(1, int(tile_w * sx))
        rh = max(1, int(tile_h * sy))
        if idx == current_index:
            qpainter.fillRect(rx, ry, rw, rh, highlight_fill)
            qpainter.setPen(highlight_pen)
            qpainter.drawRect(rx, ry, rw, rh)
            qpainter.drawText(rx + 4, max(12, ry - 4), f"Tile #{idx + 1}")
        else:
            qpainter.setPen(grid_pen)
            qpainter.drawRect(rx, ry, rw, rh)
    qpainter.end()


def compute_magnifier_source_rect(
    *,
    center_x: int,
    center_y: int,
    canvas_width: int,
    canvas_height: int,
    preview_size: int = 160,
    zoom_factor: float = 4.0,
) -> tuple[int, int, int, int]:
    """Return a clamped crop rect around the cursor for local zoom preview."""
    if canvas_width <= 0 or canvas_height <= 0 or preview_size <= 0 or zoom_factor <= 0:
        return (0, 0, 0, 0)

    crop_size = max(8, int(round(preview_size / zoom_factor)))
    crop_w = min(canvas_width, crop_size)
    crop_h = min(canvas_height, crop_size)
    left = int(round(center_x - crop_w / 2))
    top = int(round(center_y - crop_h / 2))
    left = max(0, min(left, canvas_width - crop_w))
    top = max(0, min(top, canvas_height - crop_h))
    return (left, top, crop_w, crop_h)


def compute_magnifier_anchor(
    *,
    cursor_x: int,
    cursor_y: int,
    image_width: int,
    image_height: int,
    preview_size: int = 160,
    offset: int = 18,
) -> tuple[int, int]:
    """Position the floating local zoom preview near the cursor without leaving the image."""
    if image_width <= 0 or image_height <= 0 or preview_size <= 0:
        return (0, 0)

    x = cursor_x + offset
    y = cursor_y + offset
    max_x = max(0, image_width - preview_size)
    max_y = max(0, image_height - preview_size)
    if x > max_x:
        x = cursor_x - preview_size - offset
    if y > max_y:
        y = cursor_y - preview_size - offset
    x = max(0, min(x, max_x))
    y = max(0, min(y, max_y))
    return (x, y)


def build_magnifier_preview(
    source_canvas: QPixmap | None,
    *,
    center_x: int,
    center_y: int,
    preview_size: int = 160,
    zoom_factor: float = 4.0,
) -> QPixmap | None:
    """Build a scaled local zoom preview from the current canvas around the cursor."""
    if source_canvas is None:
        return None
    if source_canvas.width() <= 0 or source_canvas.height() <= 0:
        return None

    left, top, crop_w, crop_h = compute_magnifier_source_rect(
        center_x=center_x,
        center_y=center_y,
        canvas_width=source_canvas.width(),
        canvas_height=source_canvas.height(),
        preview_size=preview_size,
        zoom_factor=zoom_factor,
    )
    if crop_w <= 0 or crop_h <= 0:
        return None

    preview = source_canvas.copy(left, top, crop_w, crop_h).scaled(
        preview_size,
        preview_size,
        Qt.AspectRatioMode.IgnoreAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    qpainter = QPainter()
    qpainter.begin(preview)
    mid_x = preview.width() // 2
    mid_y = preview.height() // 2
    qpainter.setPen(QPen(QColor("#00BFFF"), 1))
    qpainter.drawLine(mid_x, 0, mid_x, preview.height())
    qpainter.drawLine(0, mid_y, preview.width(), mid_y)
    qpainter.setPen(QPen(QColor("#FFFFFF"), 2))
    qpainter.drawRect(0, 0, max(0, preview.width() - 1), max(0, preview.height() - 1))
    qpainter.end()
    return preview
