"""
Canvas drawing utilities for bounding boxes and paste images.
"""
from typing import Any

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


def draw_paste_images_on_canvas(canvas: QPixmap, paste_images: list) -> None:
    """Draw paste images onto canvas."""
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QImage

    qpainter = QPainter()
    qpainter.begin(canvas)
    for data in paste_images:
        img, norm_x, norm_y, norm_w, norm_h = data
        x = norm_x * canvas.width()
        y = norm_y * canvas.height()
        w = norm_w * canvas.width()
        h = norm_h * canvas.height()
        qpainter.drawImage(QRect(int(x), int(y), int(w), int(h)), img)
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
