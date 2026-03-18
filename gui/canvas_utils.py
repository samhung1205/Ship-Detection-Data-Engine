"""
Canvas drawing utilities for bounding boxes and paste images.
"""
from PyQt6.QtGui import QPainter, QPen, QColor, QPixmap


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
