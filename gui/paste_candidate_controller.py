"""
Controller for in-progress paste candidate interactions.

This keeps asset loading, thumbnail refresh, placement clicks, horizontal flip,
and candidate overlay recompute logic out of ``main_window.py`` while the
underlying transform math remains compatible with the legacy workflow.
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
from PyQt6.QtCore import QRect
from PyQt6.QtGui import QImage, QPainter, QPixmap


class PasteCandidateController:
    def __init__(
        self,
        *,
        session: Any,
        get_canvas: Callable[[], Any],
        get_origin_size: Callable[[], tuple[int, int]],
        image_canvas: Any,
        preview_label: Any,
        set_mouse_press_handler: Callable[[Any], None],
        get_adjustments: Callable[[], tuple[int, int, int, int]],
        on_prepare_paste_mode: Callable[[], None],
        on_clicked_position: Callable[[int, int], None],
        on_enable_add: Callable[[bool], None],
        on_set_adjustment_labels: Callable[[str, str, str, str], None],
        on_canvas_updated: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._get_canvas = get_canvas
        self._get_origin_size = get_origin_size
        self._image_canvas = image_canvas
        self._preview_label = preview_label
        self._set_mouse_press_handler = set_mouse_press_handler
        self._get_adjustments = get_adjustments
        self._on_prepare_paste_mode = on_prepare_paste_mode
        self._on_clicked_position = on_clicked_position
        self._on_enable_add = on_enable_add
        self._on_set_adjustment_labels = on_set_adjustment_labels
        self._on_canvas_updated = on_canvas_updated

    def load_asset(self, file_path: str) -> bool:
        image = cv2.imread(file_path, cv2.IMREAD_UNCHANGED)
        if image is None or image.ndim != 3 or image.shape[2] < 4:
            return False

        alpha_w = np.sum(image[:, :, 3], axis=0)
        alpha_w[alpha_w != 0] = 1
        alpha_h = np.sum(image[:, :, 3], axis=1)
        alpha_h[alpha_h != 0] = 1
        if not np.any(alpha_w) or not np.any(alpha_h):
            return False

        min_x = int(np.min(np.where(alpha_w == 1)))
        max_x = int(np.max(np.where(alpha_w == 1)))
        min_y = int(np.min(np.where(alpha_h == 1)))
        max_y = int(np.max(np.where(alpha_h == 1)))

        cropped = image[min_y:max_y + 1, min_x:max_x + 1, :]
        cropped = np.pad(cropped, ((1, 1), (1, 1), (0, 0)), "constant", constant_values=0)

        self._session.asset_path = file_path
        self._session.origin_pasteimg = cropped.copy()
        self._session.pasteimg = cropped
        self._session.clear_candidate()
        self._refresh_thumbnail()
        return True

    def set_horizontal_flip(self, enabled: bool) -> bool:
        origin = self._session.origin_pasteimg
        if origin is None:
            return False
        self._session.pasteimg = origin[:, ::-1, :] if enabled else origin
        self._refresh_thumbnail()
        if self._session.has_anchor:
            self.recompute_preview()
        return True

    def enter_paste_mode(self) -> None:
        self._on_prepare_paste_mode()
        self._set_mouse_press_handler(self.handle_press)

    def handle_press(self, event) -> None:
        mx = int(event.position().x())
        my = int(event.position().y())
        self._on_clicked_position(mx, my)
        canvas = self._get_canvas()
        if canvas is None:
            return
        if mx >= canvas.width() or my >= canvas.height():
            return

        self._on_prepare_paste_mode()
        self._on_enable_add(True)
        self._session.set_anchor(mx, my)
        self.recompute_preview()

    def recompute_preview(self) -> None:
        scale_value, rotation_deg, brightness, contrast = self._get_adjustments()
        rate = pow(10, (scale_value - 50) / 50)
        self._on_set_adjustment_labels(
            f"{int(100 * rate)} %",
            f"{rotation_deg} °",
            str(brightness),
            str(contrast),
        )

        candidate = self._session
        canvas = self._get_canvas()
        origin_width, origin_height = self._get_origin_size()
        if (
            canvas is None
            or candidate.pasteimg is None
            or not candidate.has_anchor
            or origin_width <= 0
            or origin_height <= 0
        ):
            return

        try:
            width = int(candidate.pasteimg_width * rate)
            height = int(candidate.pasteimg_height * rate)
            dim = (width, height)
            candidate.resizeimg = cv2.resize(
                candidate.pasteimg,
                dim,
                interpolation=cv2.INTER_AREA,
            )
            r_h, r_w = candidate.resizeimg.shape[:2]

            if r_h % 2 == 0:
                candidate.resizeimg = np.pad(
                    candidate.resizeimg,
                    ((1, 0), (0, 0), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                r_h += 1
            if r_w % 2 == 0:
                candidate.resizeimg = np.pad(
                    candidate.resizeimg,
                    ((0, 0), (1, 0), (0, 0)),
                    "constant",
                    constant_values=0,
                )
                r_w += 1

            center_x, center_y = (r_w // 2, r_h // 2)
            matrix = cv2.getRotationMatrix2D((center_x, center_y), -rotation_deg, 1)
            cos = np.abs(matrix[0, 0])
            sin = np.abs(matrix[0, 1])
            n_w = int((r_h * sin) + (r_w * cos))
            n_h = int((r_h * cos) + (r_w * sin))
            matrix[0, 2] += (n_w - r_w) / 2
            matrix[1, 2] += (n_h - r_h) / 2
            candidate.rotated = cv2.warpAffine(
                candidate.resizeimg,
                matrix,
                (n_w, n_h),
                flags=cv2.INTER_AREA,
            )

            b = brightness - 100
            c = max(-99, contrast - 100)
            bc_img = candidate.rotated[:, :, :3] * (c / 100 + 1) - c + b
            bc_img = np.clip(bc_img, 0, 255)
            bc_img = np.uint8(bc_img)
            candidate.bc_image = np.dstack((bc_img, candidate.rotated[:, :, 3]))

            left = candidate.anchor_x - n_w // 2
            top = candidate.anchor_y - n_h // 2
            right = candidate.anchor_x + n_w - n_w // 2
            down = candidate.anchor_y + n_h - n_h // 2

            candidate.pasteimg_canvas = canvas.copy()
            qpainter = QPainter()
            qpainter.begin(candidate.pasteimg_canvas)
            rgba_img = cv2.cvtColor(candidate.bc_image, cv2.COLOR_BGRA2RGBA)
            bytes_per_line = 4 * n_w
            pimage = QImage(
                rgba_img,
                n_w,
                n_h,
                bytes_per_line,
                QImage.Format.Format_RGBA8888,
            )
            qpainter.drawImage(QRect(left, top, n_w, n_h), pimage)
            qpainter.end()
            self._image_canvas.paint_label_only(candidate.pasteimg_canvas)
            self._notify_canvas_updated()

            alpha_w = np.sum(candidate.bc_image[:, :, 3], axis=0)
            alpha_w[alpha_w != 0] = 1
            x1 = left + np.min(np.where(alpha_w == 1)) - 1
            x2 = right - (alpha_w.shape[0] - np.max(np.where(alpha_w == 1))) + 1
            alpha_h = np.sum(candidate.bc_image[:, :, 3], axis=1)
            alpha_h[alpha_h != 0] = 1
            y1 = top + np.min(np.where(alpha_h == 1)) - 1
            y2 = down - (alpha_h.shape[0] - np.max(np.where(alpha_h == 1))) + 1

            real_x1 = max(0, int(x1 * origin_width / candidate.pasteimg_canvas.width()))
            real_y1 = max(0, int(y1 * origin_height / candidate.pasteimg_canvas.height()))
            real_x2 = min(
                int(x2 * origin_width / candidate.pasteimg_canvas.width()),
                origin_width,
            )
            real_y2 = min(
                int(y2 * origin_height / candidate.pasteimg_canvas.height()),
                origin_height,
            )

            candidate.norm_pimg = [
                pimage,
                left / candidate.pasteimg_canvas.width(),
                top / candidate.pasteimg_canvas.height(),
                n_w / candidate.pasteimg_canvas.width(),
                n_h / candidate.pasteimg_canvas.height(),
            ]
            candidate.bbox_pimg = [
                x1,
                y1,
                x2,
                y2,
                candidate.pasteimg_canvas.width(),
                candidate.pasteimg_canvas.height(),
            ]
            candidate.real_bbox_pimg = [real_x1, real_y1, real_x2, real_y2]
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            return

    def _refresh_thumbnail(self) -> None:
        candidate = self._session
        if candidate.pasteimg is None:
            return
        pasteimg = cv2.cvtColor(candidate.pasteimg, cv2.COLOR_BGRA2RGBA)
        (
            candidate.pasteimg_height,
            candidate.pasteimg_width,
            candidate.pasteimg_channel,
        ) = candidate.pasteimg.shape
        bytes_per_line = candidate.pasteimg_channel * candidate.pasteimg_width
        pimg = QImage(
            pasteimg,
            candidate.pasteimg_width,
            candidate.pasteimg_height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888,
        )
        candidate.paste_canvas = QPixmap.fromImage(pimg)
        if candidate.pasteimg_width < candidate.pasteimg_height:
            candidate.paste_canvas = candidate.paste_canvas.scaled(
                int(80 * candidate.pasteimg_width / candidate.pasteimg_height),
                80,
            )
        else:
            candidate.paste_canvas = candidate.paste_canvas.scaled(
                80,
                int(80 * candidate.pasteimg_height / candidate.pasteimg_width),
            )
        self._preview_label.setPixmap(candidate.paste_canvas)

    def _notify_canvas_updated(self) -> None:
        if self._on_canvas_updated is not None:
            self._on_canvas_updated()
