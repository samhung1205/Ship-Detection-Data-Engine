"""
Controller for in-progress paste candidate interactions.

This keeps asset loading, thumbnail refresh, placement clicks, flip/transform,
and candidate overlay recompute logic out of ``main_window.py`` while the
underlying transform math remains compatible with the legacy workflow.
"""
from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np
from PyQt6.QtCore import QRect, Qt
from PyQt6.QtGui import QImage, QPainter, QPixmap

from sdde.augmentation import PasteAdjustments
from sdde.paste_planning import rect_contains


_ALPHA_BBOX_THRESHOLD = 16


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
        get_adjustments: Callable[[], PasteAdjustments],
        is_smart_mode_enabled: Callable[[], bool],
        get_smart_zone_rect: Callable[[], tuple[int, int, int, int] | None],
        on_prepare_paste_mode: Callable[[], None],
        on_clicked_position: Callable[[int, int], None],
        on_enable_add: Callable[[bool], None],
        on_set_adjustment_labels: Callable[[PasteAdjustments], None],
        on_set_status_message: Callable[[str], None] | None = None,
        on_canvas_updated: Callable[[], None] | None = None,
    ) -> None:
        self._session = session
        self._get_canvas = get_canvas
        self._get_origin_size = get_origin_size
        self._image_canvas = image_canvas
        self._preview_label = preview_label
        self._set_mouse_press_handler = set_mouse_press_handler
        self._get_adjustments = get_adjustments
        self._is_smart_mode_enabled = is_smart_mode_enabled
        self._get_smart_zone_rect = get_smart_zone_rect
        self._on_prepare_paste_mode = on_prepare_paste_mode
        self._on_clicked_position = on_clicked_position
        self._on_enable_add = on_enable_add
        self._on_set_adjustment_labels = on_set_adjustment_labels
        self._on_set_status_message = on_set_status_message
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
        self._session.clear_candidate()
        self._on_enable_add(False)
        self._refresh_source_from_adjustments()
        self._set_status_message("Asset loaded. Click the image to preview a paste placement.")
        return True

    def set_horizontal_flip(self, enabled: bool) -> bool:
        del enabled
        if not self._refresh_source_from_adjustments():
            return False
        if self._session.has_anchor:
            self.recompute_preview()
        return True

    def set_vertical_flip(self, enabled: bool) -> bool:
        del enabled
        if not self._refresh_source_from_adjustments():
            return False
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
        self._session.set_anchor(mx, my)
        self.recompute_preview()

    def recompute_preview(self) -> None:
        adjustments = self._get_adjustments()
        self._on_set_adjustment_labels(adjustments)
        if not self._refresh_source_from_adjustments(adjustments):
            self._on_enable_add(False)
            self._set_status_message("Load an RGBA asset first.")
            return

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
            self._on_enable_add(False)
            self._set_status_message("Click the image to choose a paste anchor.")
            return

        try:
            display_image, display_alpha = self._build_transformed_image(
                candidate.pasteimg,
                adjustments,
                scale_multiplier=1.0,
            )
            if display_image is None:
                self._on_enable_add(False)
                self._set_status_message("Unable to render the current paste preview.")
                return

            candidate.bc_image = display_image
            n_h, n_w = display_image.shape[:2]
            left = candidate.anchor_x - n_w // 2
            top = candidate.anchor_y - n_h // 2

            candidate.pasteimg_canvas = canvas.copy()
            qpainter = QPainter()
            qpainter.begin(candidate.pasteimg_canvas)
            qpainter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
            pimage = self._qimage_from_bgra(display_image)
            qpainter.drawImage(QRect(left, top, n_w, n_h), pimage)
            qpainter.end()
            self._image_canvas.paint_label_only(candidate.pasteimg_canvas)
            self._notify_canvas_updated()

            display_bbox = self._alpha_bbox_from_mask(
                display_alpha,
                left=left,
                top=top,
            )
            if display_bbox is None:
                candidate.norm_pimg = None
                candidate.bbox_pimg = None
                candidate.real_bbox_pimg = None
                self._on_enable_add(False)
                self._set_status_message("Current paste becomes empty after alpha filtering.")
                return

            export_scale = min(
                origin_width / max(1, candidate.pasteimg_canvas.width()),
                origin_height / max(1, candidate.pasteimg_canvas.height()),
            )
            export_image, export_alpha = self._build_transformed_image(
                candidate.pasteimg,
                adjustments,
                scale_multiplier=export_scale,
            )
            if export_image is None:
                candidate.norm_pimg = None
                candidate.bbox_pimg = None
                candidate.real_bbox_pimg = None
                self._on_enable_add(False)
                self._set_status_message("Unable to build export-resolution paste payload.")
                return

            export_h, export_w = export_image.shape[:2]
            export_anchor_x = int(round(candidate.anchor_x * origin_width / candidate.pasteimg_canvas.width()))
            export_anchor_y = int(round(candidate.anchor_y * origin_height / candidate.pasteimg_canvas.height()))
            export_left = export_anchor_x - export_w // 2
            export_top = export_anchor_y - export_h // 2
            real_bbox = self._alpha_bbox_from_mask(
                export_alpha,
                left=export_left,
                top=export_top,
                clamp_width=origin_width,
                clamp_height=origin_height,
            )
            if real_bbox is None:
                candidate.norm_pimg = None
                candidate.bbox_pimg = None
                candidate.real_bbox_pimg = None
                self._on_enable_add(False)
                self._set_status_message("Current paste becomes empty in the export image.")
                return

            export_qimage = self._qimage_from_bgra(export_image)
            candidate.norm_pimg = [
                pimage,
                left / candidate.pasteimg_canvas.width(),
                top / candidate.pasteimg_canvas.height(),
                n_w / candidate.pasteimg_canvas.width(),
                n_h / candidate.pasteimg_canvas.height(),
                export_qimage,
                export_left / origin_width,
                export_top / origin_height,
                export_w / origin_width,
                export_h / origin_height,
            ]
            candidate.bbox_pimg = [
                display_bbox[0],
                display_bbox[1],
                display_bbox[2],
                display_bbox[3],
                candidate.pasteimg_canvas.width(),
                candidate.pasteimg_canvas.height(),
            ]
            candidate.real_bbox_pimg = real_bbox
            can_add = True
            status_message = "Paste preview ready. Click Add to commit it."
            if self._is_smart_mode_enabled():
                smart_zone = self._get_smart_zone_rect()
                if smart_zone is None:
                    can_add = False
                    status_message = "Smart zone mode is on. Draw a valid zone before adding."
                elif not rect_contains(smart_zone, real_bbox):
                    can_add = False
                    status_message = "Paste bbox must stay inside the smart zone."
                else:
                    status_message = "Paste bbox is inside the smart zone."
            self._on_enable_add(can_add)
            self._set_status_message(status_message)
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            self._on_enable_add(False)
            self._set_status_message("Paste preview update failed.")
            return

    @staticmethod
    def _build_transformed_image(
        source: np.ndarray | None,
        adjustments: PasteAdjustments,
        *,
        scale_multiplier: float,
    ) -> tuple[np.ndarray | None, np.ndarray | None]:
        if source is None:
            return None, None

        scale_factor = max(0.01, float(adjustments.scale_factor) * float(scale_multiplier))
        src_h, src_w = source.shape[:2]
        width = max(1, int(round(src_w * scale_factor)))
        height = max(1, int(round(src_h * scale_factor)))
        resize_interp = cv2.INTER_AREA if scale_factor <= 1.0 else cv2.INTER_LINEAR
        resized = cv2.resize(source, (width, height), interpolation=resize_interp)
        r_h, r_w = resized.shape[:2]

        if r_h % 2 == 0:
            resized = np.pad(
                resized,
                ((1, 0), (0, 0), (0, 0)),
                "constant",
                constant_values=0,
            )
            r_h += 1
        if r_w % 2 == 0:
            resized = np.pad(
                resized,
                ((0, 0), (1, 0), (0, 0)),
                "constant",
                constant_values=0,
            )
            r_w += 1

        center_x, center_y = (r_w // 2, r_h // 2)
        matrix = cv2.getRotationMatrix2D(
            (center_x, center_y),
            -float(adjustments.rotation_deg),
            1,
        )
        cos = np.abs(matrix[0, 0])
        sin = np.abs(matrix[0, 1])
        n_w = int((r_h * sin) + (r_w * cos))
        n_h = int((r_h * cos) + (r_w * sin))
        matrix[0, 2] += (n_w - r_w) / 2
        matrix[1, 2] += (n_h - r_h) / 2
        rotated = cv2.warpAffine(
            resized,
            matrix,
            (n_w, n_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        b = adjustments.brightness - 100
        c = max(-99, adjustments.contrast - 100)
        rgb = rotated[:, :, :3].astype(np.float32)
        rgb = rgb * (c / 100 + 1) - c + b
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        if adjustments.blur_radius > 0:
            blur_ksize = 2 * adjustments.blur_radius + 1
            rgb = cv2.GaussianBlur(rgb, (blur_ksize, blur_ksize), 0)
        if adjustments.motion_blur_enabled and adjustments.motion_blur_length > 1:
            rgb = PasteCandidateController._apply_motion_blur(
                rgb,
                adjustments.motion_blur_length,
                adjustments.motion_blur_angle_deg,
            )

        alpha = rotated[:, :, 3].copy()
        if adjustments.feather_radius > 0:
            feather_ksize = 2 * adjustments.feather_radius + 1
            alpha = cv2.GaussianBlur(alpha, (feather_ksize, feather_ksize), 0)
        if adjustments.opacity_pct < 100:
            alpha = np.clip(
                alpha.astype(np.float32) * (adjustments.opacity_pct / 100.0),
                0,
                255,
            ).astype(np.uint8)

        bbox_alpha = np.ascontiguousarray(alpha.copy())
        rgba = np.ascontiguousarray(np.dstack((rgb, alpha)))
        if adjustments.shadow_enabled and adjustments.shadow_opacity_pct > 0 and adjustments.shadow_offset_px > 0:
            rgba = PasteCandidateController._compose_shadow(
                rgba,
                shadow_opacity_pct=adjustments.shadow_opacity_pct,
                shadow_offset_px=adjustments.shadow_offset_px,
            )

        return rgba, bbox_alpha

    @staticmethod
    def _apply_motion_blur(
        rgb: np.ndarray,
        length: int,
        angle_deg: int,
    ) -> np.ndarray:
        size = max(1, int(length))
        if size <= 1:
            return rgb
        if size % 2 == 0:
            size += 1
        kernel = np.zeros((size, size), dtype=np.float32)
        center = size // 2
        radians = np.deg2rad(float(angle_deg))
        dx = int(round(np.cos(radians) * center))
        dy = int(round(np.sin(radians) * center))
        cv2.line(
            kernel,
            (center - dx, center - dy),
            (center + dx, center + dy),
            1.0,
            1,
        )
        total = float(kernel.sum())
        if total <= 0:
            kernel[center, center] = 1.0
            total = 1.0
        kernel /= total
        blurred = cv2.filter2D(rgb, -1, kernel, borderType=cv2.BORDER_REPLICATE)
        return np.ascontiguousarray(blurred)

    @staticmethod
    def _compose_shadow(
        rgba: np.ndarray,
        *,
        shadow_opacity_pct: int,
        shadow_offset_px: int,
    ) -> np.ndarray:
        alpha = rgba[:, :, 3]
        if not np.any(alpha):
            return rgba
        offset = int(max(1, shadow_offset_px))
        shadow_matrix = np.float32([[1, 0, offset], [0, 1, offset]])
        shadow_alpha = cv2.warpAffine(
            alpha,
            shadow_matrix,
            (rgba.shape[1], rgba.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        shadow_blur_radius = max(2, offset // 2)
        shadow_ksize = 2 * shadow_blur_radius + 1
        shadow_alpha = cv2.GaussianBlur(shadow_alpha, (shadow_ksize, shadow_ksize), 0)
        shadow_alpha = np.clip(
            shadow_alpha.astype(np.float32) * (float(shadow_opacity_pct) / 100.0),
            0,
            255,
        ).astype(np.uint8)
        ship_a = rgba[:, :, 3].astype(np.float32) / 255.0
        shad_a = shadow_alpha.astype(np.float32) / 255.0
        final_a = ship_a + shad_a * (1.0 - ship_a)
        final_rgb = np.zeros_like(rgba[:, :, :3], dtype=np.float32)
        ship_rgb = rgba[:, :, :3].astype(np.float32)
        nonzero = final_a > 0
        final_rgb[nonzero] = (
            ship_rgb[nonzero] * ship_a[nonzero, None]
        ) / final_a[nonzero, None]
        out = np.dstack((
            np.clip(final_rgb, 0, 255).astype(np.uint8),
            np.clip(final_a * 255.0, 0, 255).astype(np.uint8),
        ))
        return np.ascontiguousarray(out)

    @staticmethod
    def _qimage_from_bgra(image: np.ndarray) -> QImage:
        height, width = image.shape[:2]
        rgba_img = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA)
        bytes_per_line = 4 * width
        return QImage(
            rgba_img.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGBA8888,
        ).copy()

    @staticmethod
    def _alpha_bbox_from_mask(
        alpha_mask: np.ndarray | None,
        *,
        left: int,
        top: int,
        clamp_width: int | None = None,
        clamp_height: int | None = None,
    ) -> list[int] | None:
        if alpha_mask is None:
            return None
        valid_mask = alpha_mask > _ALPHA_BBOX_THRESHOLD
        if not np.any(valid_mask):
            return None
        ys, xs = np.where(valid_mask)
        x1 = left + int(xs.min())
        x2 = left + int(xs.max()) + 1
        y1 = top + int(ys.min())
        y2 = top + int(ys.max()) + 1
        if clamp_width is not None:
            x1 = max(0, x1)
            x2 = min(x2, clamp_width)
        if clamp_height is not None:
            y1 = max(0, y1)
            y2 = min(y2, clamp_height)
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

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
        ).copy()
        candidate.paste_canvas = QPixmap.fromImage(pimg)
        preview_w = max(80, int(getattr(self._preview_label, "width", lambda: 80)()))
        preview_h = max(80, int(getattr(self._preview_label, "height", lambda: 80)()))
        if candidate.pasteimg_width < candidate.pasteimg_height:
            candidate.paste_canvas = candidate.paste_canvas.scaled(
                int(preview_h * candidate.pasteimg_width / candidate.pasteimg_height),
                preview_h,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        else:
            candidate.paste_canvas = candidate.paste_canvas.scaled(
                preview_w,
                int(preview_w * candidate.pasteimg_height / candidate.pasteimg_width),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self._preview_label.setPixmap(candidate.paste_canvas)

    def _refresh_source_from_adjustments(
        self,
        adjustments: PasteAdjustments | None = None,
    ) -> bool:
        candidate = self._session
        origin = candidate.origin_pasteimg
        if origin is None:
            return False
        adj = adjustments or self._get_adjustments()
        source = origin.copy()
        if adj.h_flip:
            source = source[:, ::-1, :]
        if adj.v_flip:
            source = source[::-1, :, :]
        candidate.pasteimg = np.ascontiguousarray(source)
        self._refresh_thumbnail()
        return True

    def _notify_canvas_updated(self) -> None:
        if self._on_canvas_updated is not None:
            self._on_canvas_updated()

    def _set_status_message(self, message: str) -> None:
        if self._on_set_status_message is not None:
            self._on_set_status_message(message)
