"""
Main application window for ImgLab and ImgBlending.
"""
import sys
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtGui import (
    QAction, QPixmap, QImage, QColor, QCloseEvent, QKeySequence,
)
from PyQt6.QtCore import Qt, QRect, QSignalBlocker, QTimer
import cv2

from .constants import (
    STYLE_BUTTON_PRIMARY,
    STYLE_BUTTON_SECONDARY,
    STYLE_BUTTON_SECONDARY_DISABLED,
    STYLE_LIST_WIDGET,
)
from .canvas_utils import build_magnifier_preview, draw_paste_images_on_canvas, draw_selection_overlay
from .annotation_actions_controller import AnnotationActionsController
from .annotation_draw_controller import AnnotationDrawController
from .annotation_edit_controller import AnnotationEditController
from .annotation_list_controller import AnnotationListController
from .annotation_preview_controller import AnnotationPreviewController
from .annotation_list_view import AnnotationListView
from .prediction_edit_controller import PredictionEditController
from .paste_candidate_controller import PasteCandidateController
from .paste_actions_controller import PasteActionsController
from .paste_preview_controller import PastePreviewController
from .annotation_workspace_controller import AnnotationWorkspaceController
from .canvas_widget import ImageCanvasWidget
from .annotation_controller import AnnotationController
from sdde.class_catalog import ClassCatalog, default_ship_catalog
from sdde.model_inference import YoloModelHandle, load_yolo_model, run_yolo_model_inference

from .class_mapping_service import load_class_catalog
from .attribute_panel import AttributePanel
from .dialogs import (
    ClassMappingDialog,
    ErrorAnalysisDialog,
    PasteEffectsDialog,
    PredictionReviewReportDialog,
    StatisticsDialog,
    ValidationDialog,
    ShowlabWindow,
    SaveimgWindow,
    SavelabWindow,
)
from .tile_panel import TilePanel
from sdde.tile import (
    TileConfig,
    TileRect,
    boundary_crossing_annotations,
    compute_tile_grid,
    find_neighbor_tile_index,
    find_tile_index_by_point,
)
from sdde.error_analysis import ERROR_FP, ErrorCase, match_gt_pred
from sdde.metadata_export import export_annotations_csv, export_annotations_json
from sdde.prediction import (
    STATUS_EDITED,
    STATUS_PREDICTED,
    filter_predictions_by_confidence,
    parse_predictions_yolo_txt,
    rename_prediction_class,
    update_prediction_geometry_from_canvas_rect,
)
from sdde.prediction_scan import (
    has_prediction_sidecar,
    load_prediction_sidecar,
)
from sdde.prediction_review import (
    PredictionReviewState,
    clone_predictions,
    initial_prediction_review_state,
    prediction_review_status,
    prediction_review_summary,
    update_prediction_review_state,
)
from sdde.prediction_review_report import scan_prediction_review_report
from sdde.prediction_review_store import (
    has_prediction_review_session,
    load_prediction_review_session,
    remove_prediction_review_session,
    save_prediction_review_session,
)
from sdde.augmentation import (
    PasteAdjustments,
    PasteRecord,
    bbox_from_legacy_paste_row,
    export_paste_records_csv,
    export_paste_records_json,
)
from sdde.project_config import (
    ProjectConfig,
    load_project_config,
    resolve_project_path,
    save_project_config,
)
from sdde.autosave import has_autosave, read_autosave, remove_autosave, write_autosave
from sdde.annotation_aggregate import (
    build_combined_annotation_records,
    combined_box_attributes,
    combined_gt_boxes,
)
from sdde.document import AnnotationDocument
from sdde.dataset_scan import scan_folder_annotation_records
from sdde.error_analysis_scan import scan_folder_error_cases
from sdde.image_browser import find_image_index, list_supported_images
from sdde.paste_candidate import PasteCandidateSession
from sdde.paste_document import PasteDocument
from sdde.paste_planning import normalize_rect, scale_hint_for_size_tag, size_tag_for_scale_factor
from sdde.import_export import import_json_label_file, import_yolo_hbb_label_file
from sdde.legacy_rows import class_mapping_from_object_list, legacy_blocks_from_annotations
from sdde.validation import scan_dataset_validation


class MyWidget(QtWidgets.QWidget):
    @staticmethod
    def _platform_shortcut(sequence: str) -> str:
        """
        Keep Qt shortcut text stable across platforms.

        On macOS, Qt renders ``Ctrl+X`` portable text as the native ``Cmd+X``
        shortcut in menus, so we intentionally keep the string unchanged here.
        """
        return sequence

    @classmethod
    def _platform_shortcuts(cls, sequences: list[str]) -> list[str]:
        return [cls._platform_shortcut(seq) for seq in sequences]

    @classmethod
    def _redo_shortcuts(cls) -> list[str]:
        if sys.platform == "darwin":
            return ["Ctrl+Shift+Z"]
        return cls._platform_shortcuts(["Ctrl+Shift+Z", "Ctrl+Y"])

    @staticmethod
    def _image_file_filter() -> str:
        return "IMAGE(*.jpg *.jpeg *.png *.gif *.bmp *.tif *.tiff)"

    def __init__(self, is_confirm_quit: bool = True):
        super().__init__()
        self.setWindowTitle('Ship Detection Data Engine')
        self.resize(1540, 930)
        self.setUpdatesEnabled(True)
        self.is_confirm_quit = is_confirm_quit
        self.object_list = []
        self._gt_document = AnnotationDocument()
        self._paste_candidate = PasteCandidateSession()
        self._paste_document = PasteDocument()
        self.imgfilePath = ''
        self._folder_image_paths: list[str] = []
        self._folder_image_index = -1
        self._prediction_folder_path: Path | None = None
        self._prediction_review_states: dict[str, PredictionReviewState] = {}
        self.predictions: list = []
        self._prediction_conf_threshold = 0.0
        self._visible_prediction_indices: list[int] = []
        self._fp_review_queue: list[ErrorCase] = []
        self._fp_review_index = -1
        self._fp_review_prediction_root: Path | None = None
        self._fp_review_conf_threshold = 0.0
        self._yolo_model_handle: YoloModelHandle | None = None
        self._yolo_model_path = ""
        self._error_overlay_enabled = False
        self._tile_grid: list[TileRect] = []
        self._tile_overview_mode = False
        self._project_config = ProjectConfig()
        self._project_config_path: Path | None = None
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._annotation_controller = AnnotationController(self)
        self._local_zoom_preview_size = 160
        self._local_zoom_factor = 4.0
        self._paste_zone_drag_start: tuple[int, int] | None = None
        self._paste_shadow_enabled = False
        self._paste_shadow_opacity_pct = 40
        self._paste_shadow_offset_px = 8
        self._paste_motion_blur_enabled = False
        self._paste_motion_blur_length = 9
        self._paste_motion_blur_angle_deg = 0
        self.ui()
        self.adjustUi()
        self._bootstrap_class_catalog()
        self._try_load_default_project_config()
        self._refresh_model_inference_ui()

    @property
    def canvas(self):
        """Current display pixmap (scaled + overlays); owned by ImageCanvasWidget."""
        return self._image_canvas.canvas

    @property
    def gt_document(self) -> AnnotationDocument:
        return self._gt_document

    @property
    def gt_list_view(self) -> AnnotationListView:
        return self._gt_list_view

    @property
    def paste_document(self) -> PasteDocument:
        return self._paste_document

    @property
    def paste_candidate(self) -> PasteCandidateSession:
        return self._paste_candidate

    @property
    def data(self) -> list[list]:
        return self._gt_document.data

    @data.setter
    def data(self, rows) -> None:
        self._gt_document.replace(data=rows)

    @property
    def real_data(self) -> list[list]:
        return self._gt_document.real_data

    @real_data.setter
    def real_data(self, rows) -> None:
        self._gt_document.replace(real_data=rows)

    @property
    def box_attributes(self) -> list[dict[str, str]]:
        return self._gt_document.box_attributes

    @box_attributes.setter
    def box_attributes(self, attrs) -> None:
        self._gt_document.replace(box_attributes=attrs)

    @property
    def pimg_data(self) -> list[list]:
        return self._paste_document.pimg_data

    @pimg_data.setter
    def pimg_data(self, rows) -> None:
        self._paste_document.replace(pimg_data=rows)

    @property
    def real_pimg_data(self) -> list[list]:
        return self._paste_document.real_pimg_data

    @real_pimg_data.setter
    def real_pimg_data(self, rows) -> None:
        self._paste_document.replace(real_pimg_data=rows)

    @property
    def paste_images(self) -> list:
        return self._paste_document.paste_images

    @paste_images.setter
    def paste_images(self, rows) -> None:
        self._paste_document.replace(paste_images=rows)

    @property
    def paste_records(self) -> list[PasteRecord]:
        return self._paste_document.paste_records

    @paste_records.setter
    def paste_records(self, rows) -> None:
        self._paste_document.replace(paste_records=rows)

    def _on_annotation_undo(self) -> None:
        self._annotation_controller.undo()

    def _on_annotation_redo(self) -> None:
        self._annotation_controller.redo()

    def _bootstrap_class_catalog(self) -> None:
        """Load classes.yaml (or defaults) so YOLO class index matches PRD mapping."""
        try:
            self.class_catalog = load_class_catalog(classes_yaml_path=self._classes_yaml_path())
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Class mapping",
                f"Failed to load classes.yaml:\n{self._classes_yaml_path()}\n\n{exc}",
            )
            self.class_catalog = default_ship_catalog()
        self.object_list = list(self.class_catalog.names_ordered())
        if self.object_list:
            self._enable_tools_after_classes_ready()

    def apply_class_catalog(self, catalog: ClassCatalog) -> None:
        """Apply edited ClassCatalog; refresh object_list and overlays if an image is open."""
        self.class_catalog = catalog
        self.object_list = list(catalog.names_ordered())
        self._enable_tools_after_classes_ready()
        if getattr(self, "origin_canvas", None) is not None:
            try:
                self.set_img_ratio()
            except (AttributeError, RuntimeError):
                pass

    def _enable_tools_after_classes_ready(self) -> None:
        """Enable tools after class mapping is ready; kept compatible with legacy flow."""
        self.btn_inputobj.setDisabled(False)
        self.action_input.setDisabled(False)
        self.btn_label.setDisabled(False)
        self.btn_paste.setDisabled(False)
        self.action_load.setDisabled(False)
        self.action_label.setDisabled(False)
        self.action_paste.setDisabled(False)
        self.btn_loadlab.setDisabled(False)
        self.btn_showlab.setDisabled(False)
        self.action_show.setDisabled(False)
        self.action_toggle_boxes.setDisabled(False)
        self.action_delete_selected.setDisabled(False)
        self.btn_loadpred.setDisabled(False)
        self.action_load_pred.setDisabled(False)
        self.action_load_pred_folder.setDisabled(False)
        self.action_clear_pred.setDisabled(False)
        self.action_accept_all_preds.setDisabled(False)
        self.action_reject_all_preds.setDisabled(False)
        self.pred_listwidget.setDisabled(False)
        self.btn_pred_accept.setDisabled(False)
        self.btn_pred_reject.setDisabled(False)
        self._refresh_prediction_review_actions()
        self._refresh_model_inference_ui()

    def _refresh_model_inference_ui(self) -> None:
        classes_ready = bool(self.object_list)
        image_ready = bool(self.imgfilePath and getattr(self, "origin_width", 0))
        can_run = classes_ready and image_ready
        model_tip = (
            f"Run loaded model: {Path(self._yolo_model_path).name}"
            if self._yolo_model_path
            else "Load a YOLO model (.pt / .onnx) and run inference on the current image."
        )
        if hasattr(self, "action_load_model"):
            self.action_load_model.setDisabled(not classes_ready)
        if hasattr(self, "action_run_model"):
            self.action_run_model.setDisabled(not can_run)
        if hasattr(self, "btn_runmodel"):
            self.btn_runmodel.setDisabled(not can_run)
            self.btn_runmodel.setToolTip(model_tip)

    def _refresh_prediction_review_actions(self) -> None:
        has_review_folder = self._prediction_folder_path is not None
        can_navigate_review = has_review_folder and bool(self._folder_image_paths)
        has_visible_predictions = bool(self._visible_prediction_indices)
        if hasattr(self, "action_next_review_image"):
            self.action_next_review_image.setDisabled(not can_navigate_review)
        if hasattr(self, "action_clear_saved_review"):
            self.action_clear_saved_review.setDisabled(not self._can_manage_prediction_review_session())
        if hasattr(self, "action_accept_all_preds"):
            self.action_accept_all_preds.setDisabled(not has_visible_predictions)
        if hasattr(self, "action_reject_all_preds"):
            self.action_reject_all_preds.setDisabled(not has_visible_predictions)
        if hasattr(self, "action_review_summary"):
            self.action_review_summary.setDisabled(not bool(self.imgfilePath))
        self._refresh_fp_review_actions()
        self._refresh_prediction_review_status_label()

    def _refresh_prediction_review_status_label(self) -> None:
        if not hasattr(self, "lbl_review_status"):
            return
        fp_status = self._fp_review_status_text()
        if self._prediction_folder_path is None or not self._folder_image_paths:
            self.lbl_review_status.setText(fp_status)
            return
        review_images = [path for path in self._folder_image_paths if self._has_review_prediction_for_image(path)]
        pending_images = [
            path
            for path in review_images
            if self._review_status_for_image(path) != "reviewed"
        ]
        current_summary = ""
        if self.imgfilePath and self._has_review_prediction_for_image(self.imgfilePath):
            state = self._prediction_review_states.get(self._review_key(self.imgfilePath))
            if state is not None:
                current_summary = f" | current {prediction_review_summary(state)}"
            else:
                current_summary = " | current pending"
        text = f"Review queue: {len(pending_images)} pending / {len(review_images)} with preds{current_summary}"
        if fp_status:
            text += f" | {fp_status}"
        self.lbl_review_status.setText(text)

    def _refresh_fp_review_actions(self) -> None:
        has_queue = bool(self._fp_review_queue)
        if hasattr(self, "action_next_fp_review"):
            self.action_next_fp_review.setDisabled(not has_queue)
        if hasattr(self, "action_clear_fp_review"):
            self.action_clear_fp_review.setDisabled(not has_queue)

    def _fp_review_status_text(self) -> str:
        if not self._fp_review_queue:
            return ""
        total = len(self._fp_review_queue)
        if 0 <= self._fp_review_index < total:
            image_name = Path(self._fp_review_queue[self._fp_review_index].image_id).name
            return f"FP review: {self._fp_review_index + 1}/{total} {image_name}"
        return f"FP review: {total} queued"

    def _try_load_default_project_config(self) -> None:
        """Load project_config.yaml from CWD if it exists."""
        cfg_path = Path("project_config.yaml")
        if cfg_path.exists():
            try:
                self._project_config = load_project_config(cfg_path)
                self._project_config_path = cfg_path.resolve()
                self._apply_project_config()
            except (OSError, ValueError):
                pass

    def _apply_project_config(self) -> None:
        """Push ProjectConfig values into UI widgets."""
        cfg = self._project_config
        try:
            self.class_catalog = load_class_catalog(classes_yaml_path=self._classes_yaml_path())
        except (OSError, ValueError, KeyError) as exc:
            QtWidgets.QMessageBox.critical(
                self,
                "Project config",
                f"Failed to load classes.yaml from project config:\n{self._classes_yaml_path()}\n\n{exc}",
            )
            self.class_catalog = default_ship_catalog()
        self.object_list = list(self.class_catalog.names_ordered())
        self._tile_panel.spin_size.setValue(cfg.tile_size)
        self._tile_panel.spin_stride.setValue(cfg.tile_stride)
        self._restart_autosave_timer()
        self._enable_tools_after_classes_ready()

    def _restart_autosave_timer(self) -> None:
        self._autosave_timer.stop()
        secs = self._project_config.autosave_seconds
        if secs > 0:
            self._autosave_timer.start(secs * 1000)

    def _do_autosave(self) -> None:
        if not self.imgfilePath or not self.real_data:
            return
        try:
            write_autosave(
                self.imgfilePath,
                self.real_data,
                self.box_attributes,
                self.object_list,
            )
            self._update_autosave_status("Autosaved")
        except OSError:
            self._update_autosave_status("Autosave failed")

    def _update_autosave_status(self, msg: str) -> None:
        if hasattr(self, "lbl_autosave_status"):
            self.lbl_autosave_status.setText(msg)

    def _refresh_folder_navigation_ui(self) -> None:
        count = len(self._folder_image_paths)
        has_folder = count > 0 and self._folder_image_index >= 0
        if hasattr(self, "action_prev_image"):
            self.action_prev_image.setDisabled(not (has_folder and self._folder_image_index > 0))
        if hasattr(self, "action_next_image"):
            self.action_next_image.setDisabled(
                not (has_folder and self._folder_image_index < count - 1)
            )
        if hasattr(self, "lbl_image_nav"):
            if has_folder:
                self.lbl_image_nav.setText(f"Folder image {self._folder_image_index + 1} / {count}")
            else:
                self.lbl_image_nav.setText("")
        self._refresh_prediction_review_actions()

    def _set_folder_images(
        self,
        image_paths: list[str],
        *,
        current_path: str | None = None,
    ) -> None:
        self._folder_image_paths = list(image_paths)
        if not image_paths:
            self._folder_image_index = -1
            self._refresh_folder_navigation_ui()
            return
        current = current_path or image_paths[0]
        idx = find_image_index(image_paths, current)
        self._folder_image_index = idx if idx >= 0 else 0
        self._refresh_folder_navigation_ui()

    def _sync_folder_images_from_path(self, image_path: str) -> None:
        folder_paths = list_supported_images(Path(image_path).parent)
        self._set_folder_images(folder_paths, current_path=image_path)

    def _load_adjacent_folder_image(self, step: int) -> None:
        next_idx = self._folder_image_index + step
        if next_idx < 0 or next_idx >= len(self._folder_image_paths):
            return
        next_path = self._folder_image_paths[next_idx]
        if self._load_image_file(next_path, ask_confirm=False):
            self._folder_image_index = next_idx
            self._refresh_folder_navigation_ui()

    def _check_autosave_recovery(self, image_path: str) -> None:
        """If autosave data exists for this image, offer to restore."""
        if not has_autosave(image_path):
            return
        ret = QtWidgets.QMessageBox.question(
            self, "Autosave found",
            "An autosave exists for this image.\nRestore previous annotations?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
        )
        if ret != QtWidgets.QMessageBox.StandardButton.Yes:
            remove_autosave(image_path)
            return
        data = read_autosave(image_path)
        if not data:
            return
        rd = data.get("real_data", [])
        ba = data.get("box_attributes", [])
        blocks = []
        for row in rd:
            name = str(row[0])
            x1, y1, x2, y2 = int(row[1]), int(row[2]), int(row[3]), int(row[4])
            d_row = [name, x1, y1, x2, y2, self.origin_width, self.origin_height]
            r_row = [name, x1, y1, x2, y2]
            blocks.append((d_row, r_row, name))
        if blocks:
            self._gt_actions.append_blocks(blocks)
        if ba:
            self.gt_document.apply_box_attributes(ba)
        remove_autosave(image_path)
        self._update_autosave_status("Restored from autosave")

    def open_project_config(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Open project config",
            str(self._project_config_base_dir()),
            "YAML (*.yaml *.yml)",
        )
        if not fp:
            return
        try:
            self._project_config = load_project_config(fp)
            self._project_config_path = Path(fp).resolve()
            self._apply_project_config()
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def save_project_config_as(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save project config",
            str(self._project_config_base_dir() / "project_config.yaml"),
            "YAML (*.yaml *.yml)",
        )
        if not fp:
            return
        cfg = self._project_config
        cfg.tile_size = self._tile_panel.tile_size()
        cfg.tile_stride = self._tile_panel.tile_stride()
        if self.imgfilePath:
            cfg.add_recent_image(self.imgfilePath)
        try:
            save_project_config(cfg, fp)
            self._project_config_path = Path(fp).resolve()
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def export_annotation_metadata_json(self) -> None:
        recs = self._combined_annotation_records()
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save metadata JSON", "annotations_meta.json", "JSON (*.json)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_annotations_json(recs), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def export_annotation_metadata_csv(self) -> None:
        recs = self._combined_annotation_records()
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save metadata CSV", "annotations_meta.csv", "CSV (*.csv)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_annotations_csv(recs), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def ui(self):
        left_x = 10
        left_w = 112
        left_panel_x = 4
        left_panel_w = 144
        center_x = 153
        center_w = 850
        right_x = 1010
        right_w = 510
        right_right = right_x + right_w
        paste_top = 440
        self._left_col_x = left_x
        self._left_col_width = left_w
        self._left_panel_x = left_panel_x
        self._left_panel_width = left_panel_w
        self._center_x = center_x
        self._center_width = center_w
        self._right_col_x = right_x
        self._right_col_width = right_w
        self._right_col_right = right_right
        self._paste_top = paste_top

        ### 畫布（捲動、縮放重繪、bbox / paste 疊圖；滑鼠事件由 hook 轉回 MyWidget）###
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(center_x, 30, center_w, 480)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)

        self._image_canvas = ImageCanvasWidget(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self._image_canvas)
        self.pmap = self._image_canvas.image_label
        self.pmap.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        ### control zoom ###
        self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_in.setGeometry(QRect(center_x + 30, 520, 89, 25))
        self.btn_zoom_in.setText("zoom_in")
        self.btn_zoom_in.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_in.setDisabled(True)
        self.btn_zoom_in.clicked.connect(self.set_zoom_in)

        self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.slider_zoom.setGeometry(QRect(center_x + 130, 518, 231, 28))
        self.slider_zoom.setProperty("value", 50)
        self.slider_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.slider_zoom.setDisabled(True)
        self.slider_zoom.valueChanged.connect(self.getslidervalue)

        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QRect(center_x + 490, 520, 300, 24))

        self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_out.setGeometry(QRect(center_x + 380, 520, 89, 25))
        self.btn_zoom_out.setText("zoom_out")
        self.btn_zoom_out.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_out.setDisabled(True)
        self.btn_zoom_out.clicked.connect(self.set_zoom_out)

        ### mouseMove（由 ImageCanvasWidget 轉發）###
        self.label_get_pos = QtWidgets.QLabel(self)
        self.label_get_pos.setGeometry(center_x, 552, 250, 18)
        self.label_get_pos.setText('current position = (x,y)')
        self.label_get_pos.setStyleSheet('font-size: 12px;')

        ### mousePress（由 ImageCanvasWidget 轉發；標註模式會改為 paint / paste）###
        self.label_click_pos = QtWidgets.QLabel(self)
        self.label_click_pos.setGeometry(center_x + 270, 552, 260, 18)
        self.label_click_pos.setText('clicked position = (x,y)')
        self.label_click_pos.setStyleSheet('font-size: 12px;')

        self.label_local_zoom_hint = QtWidgets.QLabel(self)
        self.label_local_zoom_hint.setGeometry(center_x + 575, 552, 275, 18)
        self.label_local_zoom_hint.setText('Hover image for local zoom preview')
        self.label_local_zoom_hint.setStyleSheet('font-size: 11px; color: #666;')

        ### show img.shape ###
        self.label_img_shape = QtWidgets.QLabel(self)
        self.label_img_shape.setGeometry(center_x, 574, center_w, 18)

        self.lbl_autosave_status = QtWidgets.QLabel(self)
        self.lbl_autosave_status.setGeometry(center_x, 596, 300, 16)
        self.lbl_autosave_status.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_autosave_status.setText("")

        self.lbl_image_nav = QtWidgets.QLabel(self)
        self.lbl_image_nav.setGeometry(center_x + 315, 596, 300, 16)
        self.lbl_image_nav.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_image_nav.setText("")

        self.lbl_review_status = QtWidgets.QLabel(self)
        self.lbl_review_status.setGeometry(center_x, 614, 620, 16)
        self.lbl_review_status.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_review_status.setText("")

        ### label_button ###
        self.btn_label = QtWidgets.QPushButton(self)
        self.btn_label.setText('Create RectBox')
        self.btn_label.setGeometry(right_x, 30, 110, 24)
        self.btn_label.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_label.setDisabled(True)
        self.btn_label.clicked.connect(self.make_label)

        ### label_list ###
        self.label_list = QtWidgets.QLabel(self)
        self.label_list.setText('Box Labels')
        self.label_list.setGeometry(right_x, 56, 200, 20)
        self.label_list.setStyleSheet('font-size: 12px;')

        self.hideBox = QtWidgets.QCheckBox(self)
        self.hideBox.move(right_x, 76)
        self.hideBox.setText('Hide Box')
        self.hideBox.clicked.connect(lambda: self.hideBbox(self.hideBox))

        self.chk_show_preds = QtWidgets.QCheckBox(self)
        self.chk_show_preds.move(right_x + 100, 76)
        self.chk_show_preds.setText('Show preds')
        self.chk_show_preds.setChecked(True)
        self.chk_show_preds.clicked.connect(self.set_img_ratio)

        pred_controls_y = 74
        pred_slider_y = 70

        self.lbl_pred_conf = QtWidgets.QLabel(self)
        self.lbl_pred_conf.setGeometry(right_x + 226, pred_controls_y, 82, 20)
        self.lbl_pred_conf.setText('Pred conf >=')
        self.lbl_pred_conf.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_pred_conf.setStyleSheet('font-size: 11px;')

        self.slider_pred_conf = QtWidgets.QSlider(self)
        self.slider_pred_conf.setOrientation(Qt.Orientation.Horizontal)
        self.slider_pred_conf.setGeometry(right_x + 304, pred_slider_y, 150, 28)
        self.slider_pred_conf.setRange(0, 100)
        self.slider_pred_conf.setValue(0)
        self.slider_pred_conf.valueChanged.connect(self._on_prediction_conf_changed)

        self.lbl_pred_conf_val = QtWidgets.QLabel(self)
        self.lbl_pred_conf_val.setGeometry(right_x + 458, pred_controls_y, 47, 20)
        self.lbl_pred_conf_val.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_pred_conf_val.setStyleSheet('font-size: 11px; color: #555;')
        self._refresh_prediction_threshold_label()

        self.listwidget = QtWidgets.QListWidget(self)
        self.listwidget.addItems([])
        self.listwidget.setGeometry(right_x, 100, right_w, 110)
        self.listwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.listwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._gt_list_view = AnnotationListView(
            count_label=self.label_list,
            list_widget=self.listwidget,
        )
        self._gt_list_view.set_total(self._gt_document.total_boxes)
        self._gt_actions = AnnotationActionsController(
            parent=self,
            command_controller=self._annotation_controller,
            list_view=self._gt_list_view,
            get_object_names=lambda: list(self.object_list),
            get_canvas=lambda: self.canvas,
            get_origin_size=lambda: (
                int(getattr(self, "origin_width", 0) or 0),
                int(getattr(self, "origin_height", 0) or 0),
            ),
            on_add_cancelled=self.set_img_ratio,
        )
        self._gt_draw = AnnotationDrawController(
            get_canvas=lambda: self.canvas,
            image_canvas=self._image_canvas,
            on_prepare_draw_mode=self._prepare_gt_draw_mode,
            on_clicked_position=self.__update_text_clicked_position,
            on_request_add_box=self._gt_actions.prompt_add_box,
            on_reset_view=self.set_img_ratio,
            on_canvas_updated=self.update,
        )
        self._gt_preview = AnnotationPreviewController(
            document=self.gt_document,
            get_canvas=lambda: self.canvas,
            image_canvas=self._image_canvas,
            on_canvas_updated=self.update,
        )
        self._gt_edit = AnnotationEditController(
            document=self.gt_document,
            list_view=self.gt_list_view,
            get_canvas=lambda: self.canvas,
            image_canvas=self._image_canvas,
            render_canvas=self._render_gt_canvas_for_interaction,
            on_request_update_box=self._request_gt_box_update,
            on_restore_preview=self._gt_preview.preview_row,
            on_canvas_updated=self.update,
        )
        self._pred_edit = PredictionEditController(
            get_selected_prediction_index=self._selected_prediction_index,
            get_prediction_canvas_rect=self._prediction_canvas_rect,
            get_canvas=lambda: self.canvas,
            image_canvas=self._image_canvas,
            render_canvas=self._render_prediction_canvas_for_interaction,
            on_request_update_prediction=self._request_prediction_update,
            on_restore_preview=self._preview_prediction_row,
            on_canvas_updated=self.update,
        )

        self._attr_panel = AttributePanel(self)
        self._attr_panel.setGeometry(right_x, 212, right_w, 196)
        self._attr_panel.set_enabled_editing(False)
        self._gt_workspace = AnnotationWorkspaceController(
            document=self.gt_document,
            list_view=self.gt_list_view,
            attr_panel=self._attr_panel,
            preview_controller=self._gt_preview,
            on_delete_row=self._gt_actions.remove_row,
            on_rename_row=self._gt_actions.rename_row,
            on_clear_all=self._gt_actions.clear_all,
        )
        self._attr_panel.values_changed.connect(self._gt_workspace.on_attr_panel_changed)
        self._attr_panel.set_recalc_size_callback(self._gt_workspace.on_recalc_size_tag)
        self._gt_list_controller = AnnotationListController(
            parent=self,
            list_view=self._gt_list_view,
            get_object_names=lambda: list(self.object_list),
            on_preview_row=self._gt_workspace.preview_current_row,
            on_row_changed=self._gt_workspace.on_row_changed,
            on_delete_row=self._gt_workspace.delete_row,
            on_rename_row=self._gt_workspace.rename_row,
            on_clear_all=self._gt_workspace.clear_all,
        )
        self.listwidget.clicked.connect(self._gt_list_controller.on_row_clicked)
        self.listwidget.currentRowChanged.connect(self._on_gt_row_changed)
        self.listwidget.customContextMenuRequested.connect(self._gt_list_controller.open_context_menu)
        self._install_default_canvas_handlers()

        self.label_clear = QtWidgets.QPushButton(self)
        self.label_clear.setText('Delete all')
        self.label_clear.setGeometry(right_right - 74, 410, 74, 24)
        self.label_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.label_clear.clicked.connect(self._gt_list_controller.confirm_clear_all)

        ### paste_button ###
        self.btn_paste = QtWidgets.QPushButton(self)
        self.btn_paste.setText('Paste Image')
        self.btn_paste.setGeometry(right_x, paste_top, 110, 24)
        self.btn_paste.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_paste.setDisabled(True)
        self.btn_paste.clicked.connect(self.pasteImg)

        self.btn_paste_effects = QtWidgets.QPushButton(self)
        self.btn_paste_effects.setGeometry(right_x, paste_top + 222, 100, 24)
        self.btn_paste_effects.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_paste_effects.clicked.connect(self._edit_paste_effects)

        self.lbl_paste_mode = QtWidgets.QLabel(self)
        self.lbl_paste_mode.setText('Mode')
        self.lbl_paste_mode.setGeometry(right_x, paste_top + 28, 42, 20)
        self.lbl_paste_mode.setStyleSheet('font-size: 12px;')

        self.combo_paste_mode = QtWidgets.QComboBox(self)
        self.combo_paste_mode.setGeometry(right_x + 42, paste_top + 27, 110, 24)
        self.combo_paste_mode.addItems(['Manual', 'Smart zone'])
        self.combo_paste_mode.currentIndexChanged.connect(self._on_paste_mode_changed)

        self.btn_set_zone = QtWidgets.QPushButton(self)
        self.btn_set_zone.setText('Set zone')
        self.btn_set_zone.setGeometry(right_x + 160, paste_top + 27, 72, 24)
        self.btn_set_zone.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_set_zone.setDisabled(True)
        self.btn_set_zone.clicked.connect(self._begin_paste_zone_selection)

        self.btn_clear_zone = QtWidgets.QPushButton(self)
        self.btn_clear_zone.setText('Clear zone')
        self.btn_clear_zone.setGeometry(right_x + 238, paste_top + 27, 76, 24)
        self.btn_clear_zone.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_clear_zone.setDisabled(True)
        self.btn_clear_zone.clicked.connect(self._clear_paste_zone)

        self.lbl_paste_target = QtWidgets.QLabel(self)
        self.lbl_paste_target.setText('Target')
        self.lbl_paste_target.setGeometry(right_x + 320, paste_top + 28, 40, 20)
        self.lbl_paste_target.setStyleSheet('font-size: 12px;')

        self.combo_paste_size = QtWidgets.QComboBox(self)
        self.combo_paste_size.setGeometry(right_x + 362, paste_top + 27, 100, 24)
        self.combo_paste_size.addItems(['small', 'medium', 'large'])
        self.combo_paste_size.setCurrentText('medium')
        self.combo_paste_size.currentTextChanged.connect(self._refresh_paste_size_hint)

        self.lbl_paste_status = QtWidgets.QLabel(self)
        self.lbl_paste_status.setGeometry(right_x, paste_top + 54, 250, 18)
        self.lbl_paste_status.setStyleSheet('font-size: 11px; color: #4A6A88;')
        self.lbl_paste_status.setText('Paste mode: manual')

        self.lbl_paste_size_hint = QtWidgets.QLabel(self)
        self.lbl_paste_size_hint.setGeometry(right_x + 250, paste_top + 54, right_w - 250, 18)
        self.lbl_paste_size_hint.setStyleSheet('font-size: 11px; color: #666;')
        self.lbl_paste_size_hint.setText('Size hint: load asset')
        self.lbl_paste_size_hint.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self._refresh_paste_effects_button()

        self.label_pasteimg = QtWidgets.QLabel(self)
        self.label_pasteimg.setText('Image')
        self.label_pasteimg.setGeometry(right_x, paste_top + 74, 80, 20)
        self.label_pasteimg.setStyleSheet('font-size: 12px;')

        self.Hflip = QtWidgets.QCheckBox(self)
        self.Hflip.move(right_x + 50, paste_top + 74)
        self.Hflip.setText('HorizontalFlip')
        self.Hflip.setDisabled(True)
        self.Hflip.clicked.connect(lambda: self.Hflippimg(self.Hflip))

        self.Vflip = QtWidgets.QCheckBox(self)
        self.Vflip.move(right_x + 180, paste_top + 74)
        self.Vflip.setText('VerticalFlip')
        self.Vflip.setDisabled(True)
        self.Vflip.clicked.connect(lambda: self.Vflippimg(self.Vflip))

        self.white_canvas = QPixmap(100, 120)
        self.white_canvas.fill(QColor('#ffffff'))
        self.pmap_pasteimg = QtWidgets.QLabel(self)
        self.pmap_pasteimg.setGeometry(right_x, paste_top + 94, 100, 120)
        self.pmap_pasteimg.setStyleSheet('border: 1px solid #D3D3D3;')
        self.pmap_pasteimg.setPixmap(self.white_canvas)

        self.btn_chooseimg = QtWidgets.QPushButton(self)
        self.btn_chooseimg.setText('Choose')
        self.btn_chooseimg.setGeometry(right_x, paste_top + 292, 50, 20)
        self.btn_chooseimg.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_chooseimg.clicked.connect(self.chooseImg)

        self.btn_add = QtWidgets.QPushButton(self)
        self.btn_add.setText('Add')
        self.btn_add.setGeometry(right_x + 55, paste_top + 292, 50, 20)
        self.btn_add.setStyleSheet(STYLE_BUTTON_SECONDARY_DISABLED)
        self.btn_add.setDisabled(True)
        self.btn_add.clicked.connect(self.inputPimg)

        self.btn_reset = QtWidgets.QPushButton(self)
        self.btn_reset.setText('Reset')
        self.btn_reset.setGeometry(right_right - 60, paste_top + 292, 60, 20)
        self.btn_reset.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_reset.clicked.connect(self.resetVal)

        ### Paste_imgae_QListWidget ###
        self.pimg_list = QtWidgets.QLabel(self)
        self.pimg_list.setText('Paste Images')
        self.pimg_list.setGeometry(right_x, paste_top + 318, 200, 20)
        self.pimg_list.setStyleSheet('font-size: 12px;')

        self.pimglistwidget = QtWidgets.QListWidget(self)
        self.pimglistwidget.addItems([])
        self.pimglistwidget.setGeometry(right_x, paste_top + 340, right_w, 74)
        self.pimglistwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.pimglistwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.pimg_clear = QtWidgets.QPushButton(self)
        self.pimg_clear.setText('Delete all')
        self.pimg_clear.setGeometry(right_right - 74, paste_top + 418, 74, 24)
        self.pimg_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)

        self._paste_actions = PasteActionsController(
            parent=self,
            document=self.paste_document,
            list_widget=self.pimglistwidget,
            count_label=self.pimg_list,
            get_object_names=lambda: list(self.object_list),
            append_object_name=self.object_list.append,
            image_canvas=self._image_canvas,
            on_canvas_updated=self.update,
            on_rows_changed=self._refresh_paste_canvas_from_committed,
            on_add_cancelled=self._image_canvas.sync_label_from_canvas,
            on_disable_add=lambda: self.btn_add.setDisabled(True),
            build_record=self._build_paste_record,
        )
        self._paste_candidate_controller = PasteCandidateController(
            session=self.paste_candidate,
            get_canvas=lambda: self.canvas,
            get_origin_size=lambda: (
                int(getattr(self, "origin_width", 0) or 0),
                int(getattr(self, "origin_height", 0) or 0),
            ),
            image_canvas=self._image_canvas,
            preview_label=self.pmap_pasteimg,
            set_mouse_press_handler=self._image_canvas.set_mouse_press_handler,
            get_adjustments=self._get_paste_adjustments,
            is_smart_mode_enabled=self._smart_paste_enabled,
            get_smart_zone_rect=lambda: self.paste_candidate.smart_zone_rect,
            on_prepare_paste_mode=self._prepare_paste_mode,
            on_clicked_position=self.__update_text_clicked_position,
            on_enable_add=lambda enabled: self.btn_add.setDisabled(not enabled),
            on_set_adjustment_labels=self._set_paste_adjustment_labels,
            on_set_status_message=self._set_paste_status_message,
            on_canvas_updated=self.update,
        )
        self._paste_preview = PastePreviewController(
            document=self.paste_document,
            get_canvas=lambda: self.canvas,
            image_canvas=self._image_canvas,
            on_canvas_updated=self.update,
        )
        self.pimglistwidget.clicked.connect(
            lambda _index: self._paste_preview.preview_row(self._paste_actions.current_row())
        )
        self.pimglistwidget.customContextMenuRequested.connect(self._paste_actions.open_context_menu)
        self.pimg_clear.clicked.connect(self._paste_actions.clear_all)

        self.mbox = QtWidgets.QMessageBox(self)

        ### menu_File ###
        self.menubar = QtWidgets.QMenuBar(self)
        self.menu_file = QtWidgets.QMenu('File')

        self.action_open_project = QAction('Open project config…')
        self.action_open_project.triggered.connect(self.open_project_config)
        self.menu_file.addAction(self.action_open_project)
        self.action_save_project = QAction('Save project config…')
        self.action_save_project.setShortcut(QKeySequence.StandardKey.SaveAs)
        self.action_save_project.triggered.connect(self.save_project_config_as)
        self.menu_file.addAction(self.action_save_project)
        self.menu_file.addSeparator()

        self.action_open = QAction('Open Image')
        self.action_open.setShortcut(QKeySequence.StandardKey.Open)
        self.action_open.triggered.connect(self.newFile)
        self.menu_file.addAction(self.action_open)

        self.action_open_folder = QAction('Open Folder…')
        self.action_open_folder.triggered.connect(self.openFolder)
        self.menu_file.addAction(self.action_open_folder)

        self.action_prev_image = QAction('Previous image')
        self.action_prev_image.setDisabled(True)
        self.action_prev_image.triggered.connect(self.open_previous_image)
        self.menu_file.addAction(self.action_prev_image)

        self.action_next_image = QAction('Next image')
        self.action_next_image.setDisabled(True)
        self.action_next_image.triggered.connect(self.open_next_image)
        self.menu_file.addAction(self.action_next_image)

        self.action_input = QAction('Class mapping')
        self.action_input.setShortcut(self._platform_shortcut('Ctrl+I'))
        self.action_input.setDisabled(True)
        self.action_input.triggered.connect(self.inputObj)
        self.menu_file.addAction(self.action_input)
        self.menu_file.addSeparator()

        self.action_load = QAction('Load Label')
        self.action_load.setShortcut(self._platform_shortcut('Ctrl+L'))
        self.action_load.setDisabled(True)
        self.action_load.triggered.connect(self.loadLabel)
        self.menu_file.addAction(self.action_load)

        self.action_load_pred = QAction('Load predictions…')
        self.action_load_pred.setDisabled(True)
        self.action_load_pred.triggered.connect(self.load_predictions)
        self.menu_file.addAction(self.action_load_pred)
        self.action_load_pred_folder = QAction('Load prediction folder…')
        self.action_load_pred_folder.setDisabled(True)
        self.action_load_pred_folder.triggered.connect(self.load_prediction_folder)
        self.menu_file.addAction(self.action_load_pred_folder)
        self.action_next_review_image = QAction('Next review image')
        self.action_next_review_image.setDisabled(True)
        self.action_next_review_image.triggered.connect(self.open_next_review_image)
        self.menu_file.addAction(self.action_next_review_image)
        self.action_clear_saved_review = QAction('Clear saved review state…')
        self.action_clear_saved_review.setDisabled(True)
        self.action_clear_saved_review.triggered.connect(self.clear_saved_prediction_review_state)
        self.menu_file.addAction(self.action_clear_saved_review)
        self.action_load_model = QAction('Load YOLO model…')
        self.action_load_model.setDisabled(True)
        self.action_load_model.triggered.connect(self.load_yolo_model_dialog)
        self.menu_file.addAction(self.action_load_model)
        self.action_run_model = QAction('Run model prediction')
        self.action_run_model.setDisabled(True)
        self.action_run_model.triggered.connect(self.run_model_predictions)
        self.menu_file.addAction(self.action_run_model)
        self.action_clear_pred = QAction('Clear predictions')
        self.action_clear_pred.setDisabled(True)
        self.action_clear_pred.triggered.connect(self.clear_predictions)
        self.menu_file.addAction(self.action_clear_pred)
        self.action_accept_all_preds = QAction('Accept all visible predictions')
        self.action_accept_all_preds.setDisabled(True)
        self.action_accept_all_preds.triggered.connect(self.accept_all_visible_predictions)
        self.menu_file.addAction(self.action_accept_all_preds)
        self.action_reject_all_preds = QAction('Reject all visible predictions')
        self.action_reject_all_preds.setDisabled(True)
        self.action_reject_all_preds.triggered.connect(self.reject_all_visible_predictions)
        self.menu_file.addAction(self.action_reject_all_preds)
        self.menu_file.addSeparator()

        self.action_saveimg = QAction('Save Image')
        self.action_saveimg.setDisabled(True)
        self.action_saveimg.triggered.connect(self.saveFile)
        self.menu_file.addAction(self.action_saveimg)

        self.action_savelab = QAction('Save Label')
        self.action_savelab.setShortcut('S')
        self.action_savelab.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_savelab.setDisabled(True)
        self.action_savelab.triggered.connect(self.saveLabel)
        self.menu_file.addAction(self.action_savelab)
        self.menu_file.addSeparator()
        self.action_export_meta_json = QAction('Export annotation metadata (JSON)…')
        self.action_export_meta_json.triggered.connect(self.export_annotation_metadata_json)
        self.menu_file.addAction(self.action_export_meta_json)
        self.action_export_meta_csv = QAction('Export annotation metadata (CSV)…')
        self.action_export_meta_csv.triggered.connect(self.export_annotation_metadata_csv)
        self.menu_file.addAction(self.action_export_meta_csv)
        self.menu_file.addSeparator()
        self.action_export_paste_json = QAction('Export paste metadata (JSON)…')
        self.action_export_paste_json.triggered.connect(self.export_paste_metadata_json)
        self.menu_file.addAction(self.action_export_paste_json)
        self.action_export_paste_csv = QAction('Export paste metadata (CSV)…')
        self.action_export_paste_csv.triggered.connect(self.export_paste_metadata_csv)
        self.menu_file.addAction(self.action_export_paste_csv)
        self.menubar.addMenu(self.menu_file)

        ### menu_Edit ###
        self.menu_edit = QtWidgets.QMenu('Edit')
        self.action_label = QAction('Create RectBox')
        self.action_label.setShortcut('W')
        self.action_label.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_label.setDisabled(True)
        self.action_label.triggered.connect(self.make_label)
        self.menu_edit.addAction(self.action_label)

        self.action_delete_selected = QAction('Delete selected box')
        self.action_delete_selected.setShortcut('D')
        self.action_delete_selected.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_delete_selected.setDisabled(True)
        self.action_delete_selected.triggered.connect(self._delete_selected_gt_box)
        self.menu_edit.addAction(self.action_delete_selected)

        self.action_paste = QAction('Paste Image')
        self.action_paste.setDisabled(True)
        self.action_paste.triggered.connect(self.pasteImg)
        self.menu_edit.addAction(self.action_paste)
        self.menu_edit.addSeparator()

        self.action_show = QAction('Show Label')
        self.action_show.setDisabled(True)
        self.action_show.triggered.connect(self.showLabel)
        self.menu_edit.addAction(self.action_show)

        self.action_toggle_boxes = QAction('Hide/Show boxes')
        self.action_toggle_boxes.setShortcut('H')
        self.action_toggle_boxes.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_toggle_boxes.setDisabled(True)
        self.action_toggle_boxes.triggered.connect(self._toggle_hide_boxes)
        self.menu_edit.addAction(self.action_toggle_boxes)
        self.menu_edit.addSeparator()

        self.action_tile_left = QAction('Tile left')
        self.action_tile_left.setShortcut('Alt+Left')
        self.action_tile_left.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_tile_left.triggered.connect(lambda: self._on_tile_step_requested(-1, 0))
        self.addAction(self.action_tile_left)

        self.action_tile_right = QAction('Tile right')
        self.action_tile_right.setShortcut('Alt+Right')
        self.action_tile_right.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_tile_right.triggered.connect(lambda: self._on_tile_step_requested(1, 0))
        self.addAction(self.action_tile_right)

        self.action_tile_up = QAction('Tile up')
        self.action_tile_up.setShortcut('Alt+Up')
        self.action_tile_up.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_tile_up.triggered.connect(lambda: self._on_tile_step_requested(0, -1))
        self.addAction(self.action_tile_up)

        self.action_tile_down = QAction('Tile down')
        self.action_tile_down.setShortcut('Alt+Down')
        self.action_tile_down.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_tile_down.triggered.connect(lambda: self._on_tile_step_requested(0, 1))
        self.addAction(self.action_tile_down)

        self.action_tile_overview = QAction('Tile overview')
        self.action_tile_overview.setShortcut('Alt+G')
        self.action_tile_overview.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.action_tile_overview.triggered.connect(self._toggle_tile_overview)
        self.addAction(self.action_tile_overview)

        self.action_undo = QAction('Undo')
        self.action_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.action_undo.triggered.connect(self._on_annotation_undo)
        self.menu_edit.addAction(self.action_undo)

        self.action_redo = QAction('Redo')
        self.action_redo.setShortcuts(self._redo_shortcuts())
        self.action_redo.triggered.connect(self._on_annotation_redo)
        self.menu_edit.addAction(self.action_redo)

        self.menubar.addMenu(self.menu_edit)

        ### menu_Analysis ###
        self.menu_analysis = QtWidgets.QMenu('Analysis')
        self.action_run_error = QAction('Run error analysis…')
        self.action_run_error.triggered.connect(self.run_error_analysis)
        self.menu_analysis.addAction(self.action_run_error)
        self.action_start_fp_review = QAction('Start FP-to-label review…')
        self.action_start_fp_review.triggered.connect(self.start_fp_to_label_review)
        self.menu_analysis.addAction(self.action_start_fp_review)
        self.action_next_fp_review = QAction('Next FP')
        self.action_next_fp_review.setDisabled(True)
        self.action_next_fp_review.triggered.connect(self.open_next_fp_review_case)
        self.menu_analysis.addAction(self.action_next_fp_review)
        self.action_clear_fp_review = QAction('Clear FP review queue')
        self.action_clear_fp_review.setDisabled(True)
        self.action_clear_fp_review.triggered.connect(self.clear_fp_review_queue)
        self.menu_analysis.addAction(self.action_clear_fp_review)
        self.action_toggle_error_overlay = QAction('Show GT/Pred IoU overlay')
        self.action_toggle_error_overlay.setCheckable(True)
        self.action_toggle_error_overlay.toggled.connect(self._toggle_error_overlay)
        self.menu_analysis.addAction(self.action_toggle_error_overlay)
        self.action_show_stats = QAction('Dataset statistics…')
        self.action_show_stats.triggered.connect(self.show_statistics)
        self.menu_analysis.addAction(self.action_show_stats)
        self.action_run_validation = QAction('Dataset QC…')
        self.action_run_validation.triggered.connect(self.run_dataset_qc)
        self.menu_analysis.addAction(self.action_run_validation)
        self.action_review_summary = QAction('Prediction review summary…')
        self.action_review_summary.setDisabled(True)
        self.action_review_summary.triggered.connect(self.show_prediction_review_summary)
        self.menu_analysis.addAction(self.action_review_summary)
        self.menubar.addMenu(self.menu_analysis)

        ### open_button ###
        self.btn_open = QtWidgets.QPushButton(self)
        self.btn_open.setText('Open Image')
        self.btn_open.setGeometry(left_x, 30, left_w, 24)
        self.btn_open.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_open.clicked.connect(self.newFile)

        self.btn_inputobj = QtWidgets.QPushButton(self)
        self.btn_inputobj.setText('Class mapping')
        self.btn_inputobj.setGeometry(left_x, 58, left_w, 24)
        self.btn_inputobj.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_inputobj.setDisabled(True)
        self.btn_inputobj.clicked.connect(self.inputObj)

        self.btn_loadlab = QtWidgets.QPushButton(self)
        self.btn_loadlab.setText('Load Label')
        self.btn_loadlab.setGeometry(left_x, 90, left_w, 24)
        self.btn_loadlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_loadlab.setDisabled(True)
        self.btn_loadlab.clicked.connect(self.loadLabel)

        self.btn_showlab = QtWidgets.QPushButton(self)
        self.btn_showlab.setText('Show Label')
        self.btn_showlab.setGeometry(left_x, 118, left_w, 24)
        self.btn_showlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_showlab.setDisabled(True)
        self.btn_showlab.clicked.connect(self.showLabel)

        self.btn_loadpred = QtWidgets.QPushButton(self)
        self.btn_loadpred.setText('Load preds')
        self.btn_loadpred.setGeometry(left_x, 150, left_w, 24)
        self.btn_loadpred.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_loadpred.setDisabled(True)
        self.btn_loadpred.clicked.connect(self.load_predictions)

        self.btn_runmodel = QtWidgets.QPushButton(self)
        self.btn_runmodel.setText('Run model')
        self.btn_runmodel.setGeometry(left_x, 178, left_w, 24)
        self.btn_runmodel.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_runmodel.setDisabled(True)
        self.btn_runmodel.clicked.connect(self.run_model_predictions)

        self.pred_listwidget = QtWidgets.QListWidget(self)
        self.pred_listwidget.setGeometry(left_x, 206, left_w, 110)
        self.pred_listwidget.setStyleSheet('QListWidget::item{font-size:11px;}')
        self.pred_listwidget.setDisabled(True)
        self.pred_listwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pred_listwidget.currentRowChanged.connect(self._on_prediction_row_changed)
        self.pred_listwidget.customContextMenuRequested.connect(self._open_prediction_context_menu)

        self.btn_pred_accept = QtWidgets.QPushButton(self)
        self.btn_pred_accept.setText('Accept')
        self.btn_pred_accept.setGeometry(left_x, 320, 54, 22)
        self.btn_pred_accept.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_pred_accept.setDisabled(True)
        self.btn_pred_accept.clicked.connect(self.accept_selected_prediction)

        self.btn_pred_reject = QtWidgets.QPushButton(self)
        self.btn_pred_reject.setText('Reject')
        self.btn_pred_reject.setGeometry(left_x + 58, 320, 54, 22)
        self.btn_pred_reject.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_pred_reject.setDisabled(True)
        self.btn_pred_reject.clicked.connect(self.reject_selected_prediction)

        self._tile_panel = TilePanel(self)
        self._tile_panel.setGeometry(left_panel_x, 350, left_panel_w, 222)
        self._tile_panel.tile_config_changed.connect(self._on_tile_config_changed)
        self._tile_panel.tile_index_changed.connect(self._on_tile_index_changed)
        self._tile_panel.tile_step_requested.connect(self._on_tile_step_requested)
        self._tile_panel.tile_view_toggled.connect(self._on_tile_view_toggled)
        self._tile_panel.tile_overview_toggled.connect(self._on_tile_overview_toggled)

        self.btn_saveimg = QtWidgets.QPushButton(self)
        self.btn_saveimg.setText('Save Image')
        self.btn_saveimg.setGeometry(left_x, 576, left_w, 24)
        self.btn_saveimg.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_saveimg.setDisabled(True)
        self.btn_saveimg.clicked.connect(self.saveFile)

        self.btn_savelab = QtWidgets.QPushButton(self)
        self.btn_savelab.setText('Save Label')
        self.btn_savelab.setGeometry(left_x, 604, left_w, 24)
        self.btn_savelab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_savelab.setDisabled(True)
        self.btn_savelab.clicked.connect(self.saveLabel)

        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('Quit')
        self.btn_close.setGeometry(left_x, 636, left_w, 24)
        self.btn_close.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_close.clicked.connect(self.closeFile)

    def adjustUi(self):
        sx = self._right_col_x + 118
        sw = 180
        paste_preview_top = self._paste_top + 94

        self.label_adj_1 = QtWidgets.QLabel(self)
        self.label_adj_1.setGeometry(sx, paste_preview_top, 70, 14)
        self.label_adj_1.setText('Resize')
        self.label_adj_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_1 = QtWidgets.QSlider(self)
        self.slider_1.setOrientation(Qt.Orientation.Horizontal)
        self.slider_1.setGeometry(sx + 80, paste_preview_top - 5, sw, 28)
        self.slider_1.setRange(0, 100)
        self.slider_1.setValue(50)
        self.slider_1.valueChanged.connect(self.controlpimg)
        self.label_val_1 = QtWidgets.QLabel(self)
        self.label_val_1.setGeometry(sx + 275, paste_preview_top - 5, 50, 28)
        self.label_val_1.setText("100 %")
        self.label_val_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_2 = QtWidgets.QLabel(self)
        self.label_adj_2.setGeometry(sx, paste_preview_top + 24, 70, 14)
        self.label_adj_2.setText('Rotate')
        self.label_adj_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_2 = QtWidgets.QSlider(self)
        self.slider_2.setOrientation(Qt.Orientation.Horizontal)
        self.slider_2.setGeometry(sx + 80, paste_preview_top + 19, sw, 28)
        self.slider_2.setRange(0, 360)
        self.slider_2.setValue(0)
        self.slider_2.valueChanged.connect(self.controlpimg)
        self.label_val_2 = QtWidgets.QLabel(self)
        self.label_val_2.setGeometry(sx + 275, paste_preview_top + 19, 50, 28)
        self.label_val_2.setText('0 °')
        self.label_val_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_3 = QtWidgets.QLabel(self)
        self.label_adj_3.setGeometry(sx, paste_preview_top + 48, 70, 14)
        self.label_adj_3.setText('Brightness')
        self.label_adj_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_3 = QtWidgets.QSlider(self)
        self.slider_3.setOrientation(Qt.Orientation.Horizontal)
        self.slider_3.setGeometry(sx + 80, paste_preview_top + 43, sw, 28)
        self.slider_3.setRange(0, 200)
        self.slider_3.setValue(100)
        self.slider_3.valueChanged.connect(self.controlpimg)
        self.label_val_3 = QtWidgets.QLabel(self)
        self.label_val_3.setGeometry(sx + 275, paste_preview_top + 43, 50, 28)
        self.label_val_3.setText('100')
        self.label_val_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_4 = QtWidgets.QLabel(self)
        self.label_adj_4.setGeometry(sx, paste_preview_top + 72, 70, 14)
        self.label_adj_4.setText('Contrast')
        self.label_adj_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_4 = QtWidgets.QSlider(self)
        self.slider_4.setOrientation(Qt.Orientation.Horizontal)
        self.slider_4.setGeometry(sx + 80, paste_preview_top + 67, sw, 28)
        self.slider_4.setRange(0, 200)
        self.slider_4.setValue(100)
        self.slider_4.valueChanged.connect(self.controlpimg)
        self.label_val_4 = QtWidgets.QLabel(self)
        self.label_val_4.setGeometry(sx + 275, paste_preview_top + 67, 50, 28)
        self.label_val_4.setText('100')
        self.label_val_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_5 = QtWidgets.QLabel(self)
        self.label_adj_5.setGeometry(sx, paste_preview_top + 96, 70, 14)
        self.label_adj_5.setText('Blur')
        self.label_adj_5.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_5 = QtWidgets.QSlider(self)
        self.slider_5.setOrientation(Qt.Orientation.Horizontal)
        self.slider_5.setGeometry(sx + 80, paste_preview_top + 91, sw, 28)
        self.slider_5.setRange(0, 12)
        self.slider_5.setValue(0)
        self.slider_5.valueChanged.connect(self.controlpimg)
        self.label_val_5 = QtWidgets.QLabel(self)
        self.label_val_5.setGeometry(sx + 275, paste_preview_top + 91, 50, 28)
        self.label_val_5.setText('0')
        self.label_val_5.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_6 = QtWidgets.QLabel(self)
        self.label_adj_6.setGeometry(sx, paste_preview_top + 120, 70, 14)
        self.label_adj_6.setText('Opacity')
        self.label_adj_6.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_6 = QtWidgets.QSlider(self)
        self.slider_6.setOrientation(Qt.Orientation.Horizontal)
        self.slider_6.setGeometry(sx + 80, paste_preview_top + 115, sw, 28)
        self.slider_6.setRange(0, 100)
        self.slider_6.setValue(100)
        self.slider_6.valueChanged.connect(self.controlpimg)
        self.label_val_6 = QtWidgets.QLabel(self)
        self.label_val_6.setGeometry(sx + 275, paste_preview_top + 115, 50, 28)
        self.label_val_6.setText('100 %')
        self.label_val_6.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_7 = QtWidgets.QLabel(self)
        self.label_adj_7.setGeometry(sx, paste_preview_top + 144, 70, 14)
        self.label_adj_7.setText('Feather')
        self.label_adj_7.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_7 = QtWidgets.QSlider(self)
        self.slider_7.setOrientation(Qt.Orientation.Horizontal)
        self.slider_7.setGeometry(sx + 80, paste_preview_top + 139, sw, 28)
        self.slider_7.setRange(0, 12)
        self.slider_7.setValue(0)
        self.slider_7.valueChanged.connect(self.controlpimg)
        self.label_val_7 = QtWidgets.QLabel(self)
        self.label_val_7.setGeometry(sx + 275, paste_preview_top + 139, 50, 28)
        self.label_val_7.setText('0')
        self.label_val_7.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def getslidervalue(self):
        self.set_slider_value(self.slider_zoom.value() + 1)

    def _tile_overview_active(self) -> bool:
        return self._tile_panel.is_enabled() and self._tile_overview_mode and bool(self._tile_grid)

    def _toggle_tile_overview(self) -> None:
        if not self._tile_panel.is_enabled() or not self._tile_grid:
            return
        self._tile_panel.set_overview_enabled(not self._tile_panel.overview_enabled())

    def _tile_grid_rects(self) -> list[tuple[int, int, int, int]] | None:
        if not self._tile_overview_active():
            return None
        return [(tile.x, tile.y, tile.w, tile.h) for tile in self._tile_grid]

    def _tile_index_from_canvas_point(self, mx: int, my: int) -> int | None:
        canvas = self.canvas
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if canvas is None or ow <= 0 or oh <= 0:
            return None
        cw = canvas.width()
        ch = canvas.height()
        if cw <= 0 or ch <= 0:
            return None
        gx = float(mx) * ow / float(cw)
        gy = float(my) * oh / float(ch)
        return find_tile_index_by_point(self._tile_grid, gx, gy)

    def _select_tile_index(self, index: int, *, exit_overview: bool = False) -> bool:
        if index < 0 or index >= len(self._tile_grid):
            return False
        self._tile_panel.set_current_index(index)
        if exit_overview and self._tile_overview_mode:
            self._tile_panel.set_overview_enabled(False)
            return True
        if self._tile_panel.is_enabled() and getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()
        self._refresh_tile_navigation_state()
        return True

    def _refresh_tile_navigation_state(self) -> None:
        if not self._tile_grid or not self._tile_panel.is_enabled():
            self._tile_panel.set_step_enabled(up=False, down=False, left=False, right=False)
            self.action_tile_left.setEnabled(False)
            self.action_tile_right.setEnabled(False)
            self.action_tile_up.setEnabled(False)
            self.action_tile_down.setEnabled(False)
            self.action_tile_overview.setEnabled(False)
            return
        current = self._current_tile()
        if current is None:
            self._tile_panel.set_step_enabled(up=False, down=False, left=False, right=False)
            self.action_tile_left.setEnabled(False)
            self.action_tile_right.setEnabled(False)
            self.action_tile_up.setEnabled(False)
            self.action_tile_down.setEnabled(False)
            self.action_tile_overview.setEnabled(False)
            return
        up_enabled = find_neighbor_tile_index(self._tile_grid, current_index=current.index, delta_row=-1) is not None
        down_enabled = find_neighbor_tile_index(self._tile_grid, current_index=current.index, delta_row=1) is not None
        left_enabled = find_neighbor_tile_index(self._tile_grid, current_index=current.index, delta_col=-1) is not None
        right_enabled = find_neighbor_tile_index(self._tile_grid, current_index=current.index, delta_col=1) is not None
        self._tile_panel.set_step_enabled(
            up=up_enabled,
            down=down_enabled,
            left=left_enabled,
            right=right_enabled,
        )
        self.action_tile_left.setEnabled(left_enabled)
        self.action_tile_right.setEnabled(right_enabled)
        self.action_tile_up.setEnabled(up_enabled)
        self.action_tile_down.setEnabled(down_enabled)
        self.action_tile_overview.setEnabled(True)

    def _current_tile_rect(self) -> tuple[int, int, int, int] | None:
        if self._tile_overview_active():
            return None
        tile = self._current_tile()
        if tile is None:
            return None
        return (tile.x, tile.y, tile.w, tile.h)

    def _current_tile(self) -> TileRect | None:
        if not self._tile_panel.is_enabled() or not self._tile_grid:
            return None
        idx = self._tile_panel.current_index()
        if idx < 0 or idx >= len(self._tile_grid):
            return None
        return self._tile_grid[idx]

    def _refresh_tile_boundary_hint(self) -> None:
        rows = self._current_tile_boundary_indices()
        if self._current_tile() is None:
            self._tile_panel.set_boundary_hint(enabled=False, count=0)
            return
        row_labels = [str(i + 1) for i in rows[:4]]
        tooltip = (
            "Boundary-crossing GT rows in current tile: "
            + ", ".join(f"#{i + 1} {self.real_data[i][0]}" for i in rows)
        ) if rows else ""
        self._tile_panel.set_boundary_hint(
            enabled=True,
            count=len(rows),
            row_labels=row_labels,
            tooltip=tooltip,
        )

    def _current_tile_boundary_indices(self) -> list[int]:
        tile = self._current_tile()
        if tile is None:
            return []
        return boundary_crossing_annotations(tile, self.real_data)

    def set_img_ratio(
        self,
        *,
        bbox_data_override: list | None = None,
        predictions_override: list | None = None,
    ):
        ow = int(getattr(self, "origin_width", 0) or 0)
        boundary_indices = self._current_tile_boundary_indices()
        overview_active = self._tile_overview_active()
        visible_predictions = self._filtered_predictions() if predictions_override is None else predictions_override
        self.ratio_rate, self.qpixmap_height = self._image_canvas.redraw_scaled_overlay(
            origin_canvas=self.origin_canvas,
            ratio_value=self.ratio_value,
            origin_height=self.origin_height,
            origin_width=ow,
            hide_boxes=self.hideBox.isChecked(),
            bbox_data=self.data if bbox_data_override is None else bbox_data_override,
            pimg_data=self.pimg_data,
            paste_images=self.paste_images,
            paste_zone_rect=self.paste_candidate.smart_zone_rect,
            predictions=visible_predictions,
            show_predictions=self.chk_show_preds.isChecked(),
            tile_rect=self._current_tile_rect(),
            tile_grid_rects=self._tile_grid_rects(),
            tile_grid_current_index=self._tile_panel.current_index(),
            boundary_rows=[] if overview_active else [self.real_data[i] for i in boundary_indices],
            boundary_labels=[] if overview_active else [f"#{i + 1}" for i in boundary_indices],
            error_cases=self._current_error_overlay_cases(visible_predictions),
            error_gt_boxes=self._combined_gt_boxes(),
        )
        self._image_canvas.hide_magnifier()
        self.update()
        self.__update_text_ratio()
        self.__update_text_img_shape()
        self._refresh_tile_boundary_hint()
        self._refresh_tile_navigation_state()
        self._refresh_paste_size_hint()
        self._refresh_paste_zone_controls()
        if self.paste_candidate.has_anchor and self.paste_candidate.pasteimg is not None:
            self._paste_candidate_controller.recompute_preview()
        else:
            self._refresh_paste_zone_status()

    def __update_text_ratio(self):
        self.label_ratio.setText(f"{int(100 * self.ratio_rate)} %")

    def __update_text_get_position(self, x, y):
        self.label_get_pos.setText(f'Current position = ({x}, {y})')

    def set_zoom_in(self):
        self.ratio_value = max(0, self.ratio_value - 1)
        self.set_img_ratio()

    def set_zoom_out(self):
        self.ratio_value = min(100, self.ratio_value + 1)
        self.set_img_ratio()

    def set_slider_value(self, value):
        self.ratio_value = value
        self.set_img_ratio()

    def get_position(self, event):
        mx = int(event.position().x())
        my = int(event.position().y())
        try:
            if mx < self.canvas.width() and my < self.canvas.height():
                self.__update_text_get_position(mx, my)
                self._update_local_zoom_preview(mx, my)
            else:
                self._image_canvas.hide_magnifier()
        except (AttributeError, RuntimeError):
            self.__update_text_get_position(mx, my)
            self._image_canvas.hide_magnifier()

    def _update_local_zoom_preview(self, mx: int, my: int) -> None:
        canvas = self.canvas
        if canvas is None:
            self._image_canvas.hide_magnifier()
            return
        if mx < 0 or my < 0 or mx >= canvas.width() or my >= canvas.height():
            self._image_canvas.hide_magnifier()
            return
        preview = build_magnifier_preview(
            canvas,
            center_x=mx,
            center_y=my,
            preview_size=self._local_zoom_preview_size,
            zoom_factor=self._local_zoom_factor,
        )
        self._image_canvas.show_magnifier(
            preview,
            cursor_x=mx,
            cursor_y=my,
        )

    def __update_text_clicked_position(self, x, y):
        self.label_click_pos.setText(f'Clicked position = ({x}, {y})')

    def _install_default_canvas_handlers(self) -> None:
        self._image_canvas.set_mouse_move_handler(self._handle_default_canvas_move)
        self._image_canvas.set_mouse_press_handler(self._handle_default_canvas_press)
        self._image_canvas.set_mouse_release_handler(self._handle_default_canvas_release)
        self._image_canvas.set_mouse_leave_handler(self._handle_default_canvas_leave)

    def _handle_default_canvas_move(self, event) -> None:
        self.get_position(event)
        if self._selected_prediction_index() is not None:
            self._pred_edit.handle_move(event)
            return
        self._gt_edit.handle_move(event)

    def _handle_default_canvas_press(self, event) -> None:
        if (
            event.button() == Qt.MouseButton.LeftButton
            and self._tile_overview_active()
            and self._handle_tile_overview_press(event)
        ):
            return
        self.get_clicked_position(event)
        if self._selected_prediction_index() is not None:
            self._pred_edit.handle_press(event)
            return
        self._gt_edit.handle_press(event)

    def _handle_default_canvas_release(self, event) -> None:
        if self._selected_prediction_index() is not None:
            self._pred_edit.handle_release(event)
            return
        self._gt_edit.handle_release(event)

    def _handle_default_canvas_leave(self, _event) -> None:
        self._image_canvas.hide_magnifier()

    def _handle_tile_overview_press(self, event) -> bool:
        mx = int(event.position().x())
        my = int(event.position().y())
        index = self._tile_index_from_canvas_point(mx, my)
        if index is None:
            return False
        self.__update_text_clicked_position(mx, my)
        return self._select_tile_index(index, exit_overview=True)

    def _render_gt_canvas_for_interaction(self, exclude_row: int | None) -> None:
        if exclude_row is None:
            self.set_img_ratio()
            return
        self.set_img_ratio(
            bbox_data_override=[row for i, row in enumerate(self.data) if i != exclude_row]
        )

    def _request_gt_box_update(
        self,
        row: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        if row < 0 or row >= len(self.data):
            return False
        return self._gt_actions.update_box_from_rect(
            row=row,
            item=str(self.data[row][0]),
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            select_row=True,
        )

    def _prediction_canvas_rect(self, pred_index: int) -> tuple[float, float, float, float] | None:
        canvas = self.canvas
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if canvas is None or pred_index < 0 or pred_index >= len(self.predictions) or ow <= 0 or oh <= 0:
            return None
        pred = self.predictions[pred_index]
        scale_x = canvas.width() / ow
        scale_y = canvas.height() / oh
        return (
            float(pred.x1) * scale_x,
            float(pred.y1) * scale_y,
            float(pred.x2) * scale_x,
            float(pred.y2) * scale_y,
        )

    def _preview_prediction_row(self, pred_index: int) -> None:
        rect = self._prediction_canvas_rect(pred_index)
        if rect is None:
            self.set_img_ratio()
            return
        self._render_prediction_canvas_for_interaction(pred_index)
        canvas = self.canvas
        if canvas is None:
            return
        draw_selection_overlay(
            canvas,
            x1=int(round(rect[0])),
            y1=int(round(rect[1])),
            x2=int(round(rect[2])),
            y2=int(round(rect[3])),
            fill_color=QColor(255, 165, 0, 70),
            outline_color=QColor("#FF6600"),
        )
        self._image_canvas.sync_label_from_canvas()
        self.update()

    def _render_prediction_canvas_for_interaction(self, exclude_prediction_index: int | None) -> None:
        if exclude_prediction_index is None:
            self.set_img_ratio()
            return
        visible_predictions = [
            pred
            for idx, pred in enumerate(self.predictions)
            if idx != exclude_prediction_index
            and float(getattr(pred, "confidence", 0.0)) >= self._prediction_conf_threshold
        ]
        self.set_img_ratio(predictions_override=visible_predictions)

    def _request_prediction_update(
        self,
        pred_index: int,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> bool:
        canvas = self.canvas
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if canvas is None or ow <= 0 or oh <= 0:
            return False
        ok = update_prediction_geometry_from_canvas_rect(
            self.predictions,
            pred_index,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            canvas_width=canvas.width(),
            canvas_height=canvas.height(),
            origin_width=ow,
            origin_height=oh,
        )
        if ok:
            self._refresh_pred_listwidget()
            self._sync_current_prediction_review_state()
        return ok

    def _clear_gt_selection(self, *, redraw: bool) -> None:
        if self.listwidget.currentRow() < 0:
            return
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        blocker = QSignalBlocker(self.listwidget)
        self._gt_workspace.clear_selection()
        del blocker
        if redraw:
            self.set_img_ratio()

    def _clear_prediction_selection(self, *, redraw: bool) -> None:
        if self.pred_listwidget.currentRow() < 0:
            return
        self._pred_edit.cancel_active_drag()
        blocker = QSignalBlocker(self.pred_listwidget)
        self.pred_listwidget.setCurrentRow(-1)
        del blocker
        if redraw:
            if self.listwidget.currentRow() >= 0:
                self._gt_workspace.preview_current_row()
            else:
                self.set_img_ratio()

    def _on_gt_row_changed(self, row: int) -> None:
        if row >= 0 and self.pred_listwidget.currentRow() >= 0:
            self._clear_prediction_selection(redraw=False)
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        self._install_default_canvas_handlers()
        self._gt_list_controller.on_row_changed(row)

    def _on_prediction_row_changed(self, row: int) -> None:
        self._pred_edit.cancel_active_drag()
        self._install_default_canvas_handlers()
        if row < 0:
            if self.listwidget.currentRow() >= 0:
                self._gt_workspace.preview_current_row()
            else:
                self.set_img_ratio()
            return
        self._clear_gt_selection(redraw=False)
        pred_index = self._selected_prediction_index()
        if pred_index is None:
            self.set_img_ratio()
            return
        self._preview_prediction_row(pred_index)

    def get_clicked_position(self, event):
        mx = int(event.position().x())
        my = int(event.position().y())
        self.__update_text_clicked_position(mx, my)

    def __update_text_img_shape(self):
        current_text = f"Current img shape = ({self.canvas.width()}, {self.canvas.height()})"
        origin_text = f"Origin img shape = ({self.origin_width}, {self.origin_height})"
        self.label_img_shape.setText(current_text + "    |    " + origin_text)

    def make_label(self):
        self._clear_prediction_selection(redraw=False)
        self._gt_edit.cancel_active_drag()
        self._gt_draw.enter_draw_mode()

    def _delete_selected_gt_box(self) -> None:
        self._gt_list_controller.delete_selected()

    def _toggle_hide_boxes(self) -> None:
        self.hideBox.setChecked(not self.hideBox.isChecked())
        self.hideBbox(self.hideBox)

    def _prepare_gt_draw_mode(self) -> None:
        self.hideBox.setChecked(False)
        self.hideBbox(self.hideBox)

    def hideBbox(self, _cb) -> None:
        try:
            self.set_img_ratio()
        except (AttributeError, ZeroDivisionError):
            return

    def _refresh_pred_listwidget(self) -> None:
        selected_idx = self._selected_prediction_index()
        self.pred_listwidget.clear()
        self._visible_prediction_indices = []
        for idx, p in enumerate(self.predictions):
            if float(getattr(p, "confidence", 0.0)) < self._prediction_conf_threshold:
                continue
            st = getattr(p, "pred_status", "")
            conf = float(getattr(p, "confidence", 0.0))
            name = getattr(p, "class_name", "")
            if st == STATUS_EDITED:
                suffix = " [edited]"
            elif st == STATUS_PREDICTED:
                suffix = ""
            else:
                suffix = f" [{st}]"
            self.pred_listwidget.addItem(f"{name} {conf:.2f}{suffix}")
            self._visible_prediction_indices.append(idx)
        if selected_idx is None:
            self.pred_listwidget.setCurrentRow(-1)
            self._refresh_prediction_review_actions()
            return
        try:
            self.pred_listwidget.setCurrentRow(self._visible_prediction_indices.index(selected_idx))
        except ValueError:
            self.pred_listwidget.setCurrentRow(-1)
        self._refresh_prediction_review_actions()

    def _filtered_predictions(self) -> list:
        return filter_predictions_by_confidence(
            self.predictions,
            min_confidence=self._prediction_conf_threshold,
        )

    def _selected_prediction_index(self) -> int | None:
        row = self.pred_listwidget.currentRow()
        if row < 0 or row >= len(self._visible_prediction_indices):
            return None
        return self._visible_prediction_indices[row]

    def _rename_selected_prediction(self) -> None:
        pred_idx = self._selected_prediction_index()
        if pred_idx is None or not self.object_list:
            return
        pred = self.predictions[pred_idx]
        try:
            current = self.object_list.index(pred.class_name)
        except ValueError:
            current = 0
        item, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Change prediction class",
            "Enter object name",
            list(self.object_list),
            current,
            False,
        )
        if not ok:
            return
        if not rename_prediction_class(
            self.predictions,
            pred_idx,
            new_class_name=str(item),
            object_list=self.object_list,
        ):
            return
        self._refresh_pred_listwidget()
        self._preview_prediction_row(pred_idx)
        self._sync_current_prediction_review_state()

    def _open_prediction_context_menu(self, pos) -> None:
        item = self.pred_listwidget.itemAt(pos)
        if item is None:
            return
        row = self.pred_listwidget.row(item)
        if row >= 0 and row != self.pred_listwidget.currentRow():
            self.pred_listwidget.setCurrentRow(row)
        pred_idx = self._selected_prediction_index()
        if pred_idx is None:
            return
        menu = QtWidgets.QMenu(self.pred_listwidget)
        act_rename = menu.addAction("Change class…")
        act_reject = menu.addAction("Reject prediction")
        act = menu.exec(self.pred_listwidget.mapToGlobal(pos))
        if act == act_rename:
            self._rename_selected_prediction()
        elif act == act_reject:
            self.reject_selected_prediction()

    def _refresh_prediction_threshold_label(self) -> None:
        self.lbl_pred_conf_val.setText(f"{self._prediction_conf_threshold:.2f}")

    def _current_error_overlay_cases(self, predictions: list | None = None) -> list:
        if not self._error_overlay_enabled:
            return []
        gt_boxes = self._combined_gt_boxes()
        visible_predictions = self._filtered_predictions() if predictions is None else predictions
        if not gt_boxes or not visible_predictions:
            return []
        return match_gt_pred(
            gt_boxes,
            visible_predictions,
            image_id=self.imgfilePath or "",
        )

    def _toggle_error_overlay(self, enabled: bool) -> None:
        self._error_overlay_enabled = bool(enabled)
        if self._error_overlay_enabled:
            self.hideBox.setChecked(False)
            self.chk_show_preds.setChecked(True)
        if getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()

    def _on_prediction_conf_changed(self, value: int) -> None:
        self._prediction_conf_threshold = float(value) / 100.0
        self._refresh_prediction_threshold_label()
        self._refresh_pred_listwidget()
        try:
            self.set_img_ratio()
        except (AttributeError, RuntimeError, ZeroDivisionError):
            return

    def load_yolo_model_dialog(self) -> None:
        if not self.object_list:
            QtWidgets.QMessageBox.information(
                self, "Load YOLO model", "Prepare class mapping before loading a YOLO model."
            )
            return
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "Load YOLO model",
            "",
            "YOLO Model (*.pt *.onnx *.engine);;All Files (*)",
        )
        if not fp:
            return
        try:
            self._yolo_model_handle = load_yolo_model(fp)
            self._yolo_model_path = self._yolo_model_handle.model_path
            self._refresh_model_inference_ui()
        except (OSError, RuntimeError) as e:
            QtWidgets.QMessageBox.critical(self, "Load YOLO model", str(e))

    def run_model_predictions(self) -> None:
        if not getattr(self, "origin_width", None) or not self.imgfilePath or not self.object_list:
            QtWidgets.QMessageBox.information(
                self, "Run model prediction", "Open an image first, and ensure class mapping is loaded."
            )
            return
        if self._yolo_model_handle is None:
            self.load_yolo_model_dialog()
            if self._yolo_model_handle is None:
                return
        try:
            self.predictions = run_yolo_model_inference(
                self._yolo_model_handle,
                image_path=self.imgfilePath,
                object_list=self.object_list,
                conf_threshold=0.01,
                iou_threshold=0.7,
                max_det=300,
            )
        except (OSError, RuntimeError, ValueError, IndexError) as e:
            QtWidgets.QMessageBox.critical(self, "Run model prediction", str(e))
            return
        self.chk_show_preds.setChecked(True)
        self._refresh_pred_listwidget()
        self.set_img_ratio()
        if self.pred_listwidget.count() > 0:
            self.pred_listwidget.setCurrentRow(0)

    def load_predictions(self) -> None:
        if not getattr(self, "origin_width", None) or not self.object_list:
            QtWidgets.QMessageBox.information(
                self, "Load predictions", "Open an image first, and ensure class mapping is loaded."
            )
            return
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            directory=str(self._default_prediction_directory()),
            filter="TXT (*.txt)",
        )
        if not fp:
            return
        try:
            self.predictions = self._load_predictions_from_txt_path(Path(fp))
        except (OSError, ValueError, IndexError) as e:
            QtWidgets.QMessageBox.critical(self, "Load predictions", str(e))
            return
        self._refresh_pred_listwidget()
        self.set_img_ratio()
        self._sync_current_prediction_review_state()

    def load_prediction_folder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select prediction folder",
            str(self._default_prediction_directory()),
        )
        if not folder:
            return
        new_folder = Path(folder)
        previous_folder = self._prediction_folder_path
        previous_states = dict(self._prediction_review_states)
        if self._prediction_folder_path is None or new_folder.resolve() != self._prediction_folder_path.resolve():
            self._prediction_review_states.clear()
        self._prediction_folder_path = new_folder
        session_mode = self._prompt_prediction_review_resume_mode()
        if session_mode is None:
            self._prediction_folder_path = previous_folder
            self._prediction_review_states = previous_states
            return
        if session_mode == "fresh":
            self._prediction_review_states.clear()
            self._clear_saved_prediction_review_session()
        else:
            self._prediction_review_states = self._load_saved_prediction_review_session()
        self._refresh_prediction_review_actions()
        self._auto_load_predictions_for_current_image()

    def clear_predictions(self) -> None:
        self.predictions.clear()
        self._refresh_pred_listwidget()
        self.set_img_ratio()

    def accept_selected_prediction(self) -> None:
        row = self.pred_listwidget.currentRow()
        pred_idx = self._selected_prediction_index()
        if pred_idx is None:
            return
        pred = self.predictions[pred_idx]
        if pred.pred_status not in (STATUS_PREDICTED, STATUS_EDITED):
            return
        payload = self._gt_actions.build_add_box_from_prediction(pred)
        if payload is None:
            return
        self.predictions.pop(pred_idx)
        self._refresh_pred_listwidget()
        n = self.pred_listwidget.count()
        if n:
            self.pred_listwidget.setCurrentRow(min(row, n - 1))
        else:
            self.pred_listwidget.setCurrentRow(-1)
        self._gt_actions.add_box(*payload)
        self._sync_current_prediction_review_state(accepted_delta=1)
        self._maybe_advance_prediction_review()
        # Redraw via AddBoxCommand._refresh_canvas (no duplicate orange pred).

    def accept_all_visible_predictions(self) -> None:
        visible_indices = list(self._visible_prediction_indices)
        if not visible_indices:
            return
        blocks: list[tuple[list, list, str]] = []
        accepted_indices: list[int] = []
        for pred_idx in visible_indices:
            pred = self.predictions[pred_idx]
            if pred.pred_status not in (STATUS_PREDICTED, STATUS_EDITED):
                continue
            payload = self._gt_actions.build_add_box_from_prediction(pred)
            if payload is None:
                continue
            data_row, real_row, label_text, _extended = payload
            blocks.append((data_row, real_row, label_text))
            accepted_indices.append(pred_idx)
        if not blocks:
            return
        for pred_idx in sorted(accepted_indices, reverse=True):
            self.predictions.pop(pred_idx)
        self._refresh_pred_listwidget()
        self.pred_listwidget.setCurrentRow(-1)
        self._gt_actions.append_blocks(blocks)
        self._sync_current_prediction_review_state(accepted_delta=len(accepted_indices))
        self._maybe_advance_prediction_review()

    def reject_all_visible_predictions(self) -> None:
        visible_indices = list(self._visible_prediction_indices)
        if not visible_indices:
            return
        for pred_idx in sorted(visible_indices, reverse=True):
            self.predictions.pop(pred_idx)
        self._refresh_pred_listwidget()
        self.pred_listwidget.setCurrentRow(-1)
        self.set_img_ratio()
        self._sync_current_prediction_review_state(rejected_delta=len(visible_indices))
        self._maybe_advance_prediction_review()

    def reject_selected_prediction(self) -> None:
        row = self.pred_listwidget.currentRow()
        pred_idx = self._selected_prediction_index()
        if pred_idx is None:
            return
        self.predictions.pop(pred_idx)
        self._refresh_pred_listwidget()
        n = self.pred_listwidget.count()
        if n:
            self.pred_listwidget.setCurrentRow(min(row, n - 1))
        else:
            self.pred_listwidget.setCurrentRow(-1)
        self.set_img_ratio()
        self._sync_current_prediction_review_state(rejected_delta=1)
        self._maybe_advance_prediction_review()

    # --- Tile view -----------------------------------------------------------

    def _recompute_tile_grid(self) -> None:
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if ow <= 0 or oh <= 0:
            self._tile_grid = []
            self._tile_panel.set_tile_count(0)
            self._refresh_tile_boundary_hint()
            self._refresh_tile_navigation_state()
            return
        cfg = TileConfig(
            tile_size=self._tile_panel.tile_size(),
            tile_stride=self._tile_panel.tile_stride(),
        )
        self._tile_grid = compute_tile_grid(ow, oh, cfg)
        self._tile_panel.set_tile_count(len(self._tile_grid))
        self._refresh_tile_boundary_hint()
        self._refresh_tile_navigation_state()

    def _on_tile_config_changed(self) -> None:
        self._recompute_tile_grid()
        if self._tile_panel.is_enabled() and getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()

    def _on_tile_index_changed(self, _idx: int) -> None:
        if self._tile_panel.is_enabled() and getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()

    def _on_tile_step_requested(self, delta_col: int, delta_row: int) -> None:
        if not self._tile_panel.is_enabled() or not self._tile_grid:
            return
        next_idx = find_neighbor_tile_index(
            self._tile_grid,
            current_index=self._tile_panel.current_index(),
            delta_col=delta_col,
            delta_row=delta_row,
        )
        if next_idx is None:
            return
        self._select_tile_index(next_idx)

    def _on_tile_overview_toggled(self, enabled: bool) -> None:
        self._tile_overview_mode = bool(enabled and self._tile_panel.is_enabled())
        if getattr(self, "origin_canvas", None) is None:
            return
        self.set_img_ratio()

    def _on_tile_view_toggled(self, enabled: bool) -> None:
        if not enabled:
            self._tile_overview_mode = False
        if enabled:
            self._recompute_tile_grid()
        if getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()

    def run_error_analysis(self) -> None:
        scope = self._prompt_error_analysis_scope()
        if scope is None:
            return
        if scope == "project":
            prediction_root = self._prompt_prediction_folder()
            if prediction_root is None:
                return
            result = self._scan_current_project_error_cases(prediction_root)
            if not self._confirm_project_error_analysis_run(result):
                return
            if result.total_images > 0 and result.prediction_images == 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Error analysis",
                    "No matching prediction sidecars were found in the selected prediction folder. "
                    "Results will only use GT labels in the current project scope.",
                )
            if not result.cases:
                QtWidgets.QMessageBox.information(
                    self,
                    "Error analysis",
                    "No GT annotations or predictions were found in the current project scope.",
                )
                return
            dlg = ErrorAnalysisDialog(
                self,
                gt_boxes=[],
                gt_attributes=None,
                predictions=[],
                cases=list(result.cases),
                scope_label="Current project",
                detail_label=(
                    f"Images analyzed: {result.analyzed_images} / {result.total_images}  |  "
                    f"Predictions matched: {result.prediction_images}  |  "
                    f"Label root: {self._project_label_root_display()}  |  "
                    f"Prediction root: {result.prediction_root}"
                ),
            )
            dlg.exec()
            return
        if scope == "folder":
            prediction_root = self._prompt_prediction_folder()
            if prediction_root is None:
                return
            result = self._scan_current_folder_error_cases(prediction_root)
            if not self._confirm_folder_error_analysis_run(result):
                return
            if result.total_images > 0 and result.prediction_images == 0:
                QtWidgets.QMessageBox.information(
                    self,
                    "Error analysis",
                    "No matching prediction sidecars were found in the selected prediction folder. "
                    "Results will only use GT labels in the current folder scope.",
                )
            if not result.cases:
                QtWidgets.QMessageBox.information(
                    self,
                    "Error analysis",
                    "No GT annotations or predictions were found in the current folder scope.",
                )
                return
            dlg = ErrorAnalysisDialog(
                self,
                gt_boxes=[],
                gt_attributes=None,
                predictions=[],
                cases=list(result.cases),
                scope_label="Current folder",
                detail_label=(
                    f"Images analyzed: {result.analyzed_images} / {result.total_images}  |  "
                    f"Predictions matched: {result.prediction_images}  |  "
                    f"Label root: {self._folder_label_root_display()}  |  "
                    f"Prediction folder: {result.prediction_root}"
                ),
            )
            dlg.exec()
            return

        filtered_predictions = self._filtered_predictions()
        gt_boxes = self._combined_gt_boxes()
        if not gt_boxes and not filtered_predictions:
            QtWidgets.QMessageBox.information(
                self, "Error analysis", "No GT annotations or predictions loaded."
            )
            return
        dlg = ErrorAnalysisDialog(
            self,
            gt_boxes=gt_boxes,
            gt_attributes=self._combined_box_attributes(),
            predictions=filtered_predictions,
            image_id=self.imgfilePath or "",
            scope_label="Current image",
        )
        dlg.exec()

    def start_fp_to_label_review(self) -> None:
        scope = self._prompt_fp_review_scope()
        if scope is None:
            return
        prediction_root = self._prompt_prediction_folder()
        if prediction_root is None:
            return
        if scope == "project":
            result = self._scan_current_project_error_cases(prediction_root)
            if not self._confirm_project_error_analysis_run(result):
                return
        else:
            result = self._scan_current_folder_error_cases(prediction_root)
            if not self._confirm_folder_error_analysis_run(result):
                return

        fp_cases = [
            case
            for case in result.cases
            if case.error_type == ERROR_FP and case.pred_index is not None and case.image_id
        ]
        if not fp_cases:
            QtWidgets.QMessageBox.information(
                self,
                "FP-to-label review",
                "No FP prediction candidates were found at the current confidence threshold.",
            )
            self.clear_fp_review_queue()
            return

        answer = QtWidgets.QMessageBox.question(
            self,
            "FP-to-label review",
            (
                f"Build FP-to-label queue with {len(fp_cases)} FP candidates?\n\n"
                f"Confidence threshold: {self._prediction_conf_threshold:.2f}\n"
                "Use Next FP to jump between candidates. Save Label after accepting corrections for an image."
            ),
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Ok:
            return

        self._sync_current_prediction_review_state()
        self._fp_review_queue = list(fp_cases)
        self._fp_review_index = 0
        self._fp_review_prediction_root = Path(prediction_root)
        self._fp_review_conf_threshold = self._prediction_conf_threshold
        self._refresh_prediction_review_actions()
        self._open_current_fp_review_case()

    def open_next_fp_review_case(self) -> None:
        if not self._fp_review_queue:
            QtWidgets.QMessageBox.information(
                self,
                "FP-to-label review",
                "No FP review queue is active.",
            )
            return
        if self._fp_review_index < 0:
            self._fp_review_index = 0
        else:
            self._fp_review_index += 1
        if self._fp_review_index >= len(self._fp_review_queue):
            self._fp_review_index = len(self._fp_review_queue) - 1
            QtWidgets.QMessageBox.information(
                self,
                "FP-to-label review",
                "Reached the end of the FP review queue.",
            )
            self._refresh_prediction_review_actions()
            return
        self._open_current_fp_review_case()

    def clear_fp_review_queue(self) -> None:
        self._fp_review_queue.clear()
        self._fp_review_index = -1
        self._fp_review_prediction_root = None
        self._refresh_prediction_review_actions()

    def _open_current_fp_review_case(self) -> None:
        if not (0 <= self._fp_review_index < len(self._fp_review_queue)):
            return
        case = self._fp_review_queue[self._fp_review_index]
        if not case.image_id:
            return
        target_path = str(Path(case.image_id))
        same_image = False
        if self.imgfilePath:
            try:
                same_image = Path(self.imgfilePath).resolve() == Path(target_path).resolve()
            except OSError:
                same_image = self.imgfilePath == target_path
        if not same_image:
            if not self._load_image_file(target_path, ask_confirm=True):
                return
            self._sync_folder_images_from_path(target_path)
            self._load_fp_review_predictions_for_current_image()
        elif not self.predictions:
            self._load_fp_review_predictions_for_current_image()
        self._restore_fp_review_threshold()
        selected = self._select_fp_review_prediction(case)
        self.chk_show_preds.setChecked(True)
        self.set_img_ratio()
        self._refresh_prediction_review_actions()
        if not selected:
            QtWidgets.QMessageBox.information(
                self,
                "FP-to-label review",
                "This FP candidate is no longer visible in the current prediction list. It may have already been accepted, rejected, or filtered by confidence.",
            )

    def _load_fp_review_predictions_for_current_image(self) -> None:
        if (
            self._fp_review_prediction_root is None
            or not self.imgfilePath
            or not getattr(self, "origin_width", None)
            or not getattr(self, "origin_height", None)
        ):
            return
        try:
            self.predictions = load_prediction_sidecar(
                self.imgfilePath,
                prediction_root=self._fp_review_prediction_root,
                object_list=self.object_list,
                image_w=self.origin_width,
                image_h=self.origin_height,
            )
        except (OSError, ValueError, IndexError):
            self.predictions = []
        self.chk_show_preds.setChecked(True)
        self._refresh_pred_listwidget()

    def _restore_fp_review_threshold(self) -> None:
        value = int(round(self._fp_review_conf_threshold * 100.0))
        value = max(0, min(100, value))
        if self.slider_pred_conf.value() != value:
            self.slider_pred_conf.setValue(value)
        else:
            self._prediction_conf_threshold = float(value) / 100.0
            self._refresh_prediction_threshold_label()
            self._refresh_pred_listwidget()

    def _select_fp_review_prediction(self, case: ErrorCase) -> bool:
        self._refresh_pred_listwidget()
        pred_idx = self._find_prediction_index_for_fp_case(case)
        if pred_idx is None:
            return False
        try:
            row = self._visible_prediction_indices.index(pred_idx)
        except ValueError:
            return False
        self.pred_listwidget.setCurrentRow(row)
        return True

    def _find_prediction_index_for_fp_case(self, case: ErrorCase) -> int | None:
        target_box = case.pred_box
        visible = list(self._visible_prediction_indices)
        if target_box is not None:
            for idx in visible:
                pred = self.predictions[idx]
                pred_box = (float(pred.x1), float(pred.y1), float(pred.x2), float(pred.y2))
                if not self._boxes_close(pred_box, target_box):
                    continue
                if case.pred_class and pred.class_name != case.pred_class:
                    continue
                if abs(float(pred.confidence) - float(case.confidence)) > 1e-3:
                    continue
                return idx
        if case.pred_index is not None and 0 <= case.pred_index < len(visible):
            return visible[case.pred_index]
        return None

    @staticmethod
    def _boxes_close(
        left: tuple[float, float, float, float],
        right: tuple[float, float, float, float],
        *,
        tolerance: float = 1e-3,
    ) -> bool:
        return all(abs(float(a) - float(b)) <= tolerance for a, b in zip(left, right))

    def show_statistics(self) -> None:
        scope = self._prompt_statistics_scope()
        if scope is None:
            return
        if scope == "project":
            project_scan = self._scan_current_project_annotations()
            if not self._confirm_project_statistics_run(project_scan):
                return
            recs = list(project_scan.records)
            total_images_override = project_scan.total_images
            labeled_images_override = project_scan.labeled_images
            scope_label = "Current project"
            detail_label = (
                f"Image root: {project_scan.folder_path}  |  "
                f"Label root: {self._project_label_root_display()}"
            )
        elif scope == "folder":
            folder_scan = self._scan_current_folder_annotations()
            if not self._confirm_folder_statistics_run(folder_scan):
                return
            recs = list(folder_scan.records)
            total_images_override = folder_scan.total_images
            labeled_images_override = folder_scan.labeled_images
            scope_label = "Current folder"
            detail_label = (
                f"Folder: {folder_scan.folder_path}  |  "
                f"Label root: {self._folder_label_root_display()}"
            )
        else:
            recs = self._combined_annotation_records()
            total_images_override = None
            labeled_images_override = None
            scope_label = "Current image"
            detail_label = ""
        if not recs:
            QtWidgets.QMessageBox.information(
                self, "Statistics", "No annotations to analyze."
            )
            return
        dlg = StatisticsDialog(
            self,
            records=recs,
            scope_label=scope_label,
            detail_label=detail_label,
            total_images_override=total_images_override,
            labeled_images_override=labeled_images_override,
        )
        dlg.exec()

    def run_dataset_qc(self) -> None:
        scope = self._prompt_validation_scope()
        if scope is None:
            return
        prediction_root = self._prompt_optional_validation_prediction_folder(scope)
        if prediction_root is False:
            return
        if scope == "project":
            result = self._scan_current_project_validation(prediction_root)
            if not self._confirm_project_validation_run(result, prediction_root):
                return
            scope_label = "Current project"
            detail_label = (
                f"Image root: {result.scope_path}  |  "
                f"Label root: {self._project_label_root_display()}  |  "
                f"Prediction root: {str(prediction_root) if prediction_root else '(not checked)'}"
            )
        else:
            result = self._scan_current_folder_validation(prediction_root)
            if not self._confirm_folder_validation_run(result, prediction_root):
                return
            scope_label = "Current folder"
            detail_label = (
                f"Image folder: {result.scope_path}  |  "
                f"Label root: {self._folder_label_root_display()}  |  "
                f"Prediction folder: {str(prediction_root) if prediction_root else '(not checked)'}"
            )
        dlg = ValidationDialog(
            self,
            result=result,
            scope_label=scope_label,
            detail_label=detail_label,
        )
        dlg.exec()

    def show_prediction_review_summary(self) -> None:
        scope = self._prompt_review_summary_scope()
        if scope is None:
            return
        prediction_root = self._prediction_folder_path
        if prediction_root is None:
            prediction_root = self._prompt_prediction_folder()
            if prediction_root is None:
                return
        if scope == "project":
            report = self._scan_current_project_review_report(prediction_root)
            if not self._confirm_project_review_summary_run(report):
                return
            scope_label = "Current project"
            detail_label = (
                f"Image root: {report.scope_path}  |  "
                f"Prediction root: {report.prediction_root}"
            )
        else:
            report = self._scan_current_folder_review_report(prediction_root)
            if not self._confirm_folder_review_summary_run(report):
                return
            scope_label = "Current folder"
            detail_label = (
                f"Image folder: {report.scope_path}  |  "
                f"Prediction folder: {report.prediction_root}"
            )
        dlg = PredictionReviewReportDialog(
            self,
            report=report,
            scope_label=scope_label,
            detail_label=detail_label,
        )
        dlg.exec()

    def chooseImg(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='rembg_img', filter='IMAGE(*.jpg *.png *.gif *.bmp)'
        )
        if filePath:
            self.resetVal()
            self.Hflip.setDisabled(False)
            self.Vflip.setDisabled(False)
            self.Hflip.setChecked(False)
            self.Vflip.setChecked(False)
            self._paste_candidate_controller.load_asset(filePath)
            self._refresh_paste_zone_controls()
            self._refresh_paste_size_hint()

    def Hflippimg(self, cb):
        self._paste_candidate_controller.set_horizontal_flip(cb.isChecked())
        self._refresh_paste_size_hint()

    def Vflippimg(self, cb):
        self._paste_candidate_controller.set_vertical_flip(cb.isChecked())
        self._refresh_paste_size_hint()

    def inputPimg(self):
        candidate = self.paste_candidate
        self._paste_actions.prompt_add_candidate(
            bbox_row=candidate.bbox_pimg,
            real_bbox_row=candidate.real_bbox_pimg,
            paste_image=candidate.norm_pimg,
            preview_canvas=candidate.pasteimg_canvas,
        )
        candidate.clear_candidate()
        self._refresh_paste_zone_status()

    def _refresh_paste_canvas_from_committed(self) -> None:
        try:
            pm = self.origin_canvas.scaledToHeight(
                self.qpixmap_height,
                Qt.TransformationMode.SmoothTransformation,
            )
            draw_paste_images_on_canvas(pm, self.paste_images)
            self._image_canvas.set_canvas(pm)
            self.update()
            self.hideBox.setChecked(False)
            self.hideBbox(self.hideBox)
        except (AttributeError, ZeroDivisionError):
            return

    def build_export_image_pixmap(self) -> QPixmap | None:
        origin_canvas = getattr(self, "origin_canvas", None)
        if origin_canvas is None:
            return None
        export_canvas = origin_canvas.copy()
        draw_paste_images_on_canvas(
            export_canvas,
            self.paste_images,
            prefer_export_geometry=True,
        )
        return export_canvas

    def _build_paste_record(
        self,
        class_name: str,
        real_row: list[object],
    ) -> PasteRecord | None:
        try:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_from_legacy_paste_row(real_row)
            adjustments = self._get_paste_adjustments()
            return PasteRecord(
                image_path=self.imgfilePath or "",
                asset_path=self.paste_candidate.asset_path,
                class_name=class_name,
                scale=round(adjustments.scale_factor, 4),
                rotation_deg=float(adjustments.rotation_deg),
                h_flip=adjustments.h_flip,
                v_flip=adjustments.v_flip,
                brightness=adjustments.brightness,
                contrast=adjustments.contrast,
                blur_radius=adjustments.blur_radius,
                opacity_pct=adjustments.opacity_pct,
                feather_radius=adjustments.feather_radius,
                shadow_enabled=adjustments.shadow_enabled,
                shadow_opacity_pct=adjustments.shadow_opacity_pct,
                shadow_offset_px=adjustments.shadow_offset_px,
                motion_blur_enabled=adjustments.motion_blur_enabled,
                motion_blur_length=adjustments.motion_blur_length,
                motion_blur_angle_deg=adjustments.motion_blur_angle_deg,
                bbox_x1=bbox_x1,
                bbox_y1=bbox_y1,
                bbox_x2=bbox_x2,
                bbox_y2=bbox_y2,
            )
        except (AttributeError, IndexError, ValueError):
            return None

    def export_paste_metadata_json(self) -> None:
        if not self.paste_records:
            QtWidgets.QMessageBox.information(self, "Export", "No paste records to export.")
            return
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save paste metadata JSON", "paste_meta.json", "JSON (*.json)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_paste_records_json(self.paste_records), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def export_paste_metadata_csv(self) -> None:
        if not self.paste_records:
            QtWidgets.QMessageBox.information(self, "Export", "No paste records to export.")
            return
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save paste metadata CSV", "paste_meta.csv", "CSV (*.csv)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_paste_records_csv(self.paste_records), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def resetVal(self):
        self.slider_1.setValue(50)
        self.slider_2.setValue(0)
        self.slider_3.setValue(100)
        self.slider_4.setValue(100)
        self.slider_5.setValue(0)
        self.slider_6.setValue(100)
        self.slider_7.setValue(0)
        self.Hflip.setChecked(False)
        self.Vflip.setChecked(False)
        self._reset_paste_effects()
        self._set_paste_adjustment_labels(self._get_paste_adjustments())
        self._refresh_paste_size_hint()

    def _get_paste_adjustments(self) -> PasteAdjustments:
        return PasteAdjustments(
            scale_slider=self.slider_1.value(),
            rotation_deg=self.slider_2.value(),
            brightness=self.slider_3.value(),
            contrast=self.slider_4.value(),
            blur_radius=self.slider_5.value(),
            opacity_pct=self.slider_6.value(),
            feather_radius=self.slider_7.value(),
            h_flip=self.Hflip.isChecked(),
            v_flip=self.Vflip.isChecked(),
            shadow_enabled=self._paste_shadow_enabled,
            shadow_opacity_pct=self._paste_shadow_opacity_pct,
            shadow_offset_px=self._paste_shadow_offset_px,
            motion_blur_enabled=self._paste_motion_blur_enabled,
            motion_blur_length=self._paste_motion_blur_length,
            motion_blur_angle_deg=self._paste_motion_blur_angle_deg,
        )

    def _set_paste_adjustment_labels(self, adjustments: PasteAdjustments) -> None:
        self.label_val_1.setText(f"{int(round(100 * adjustments.scale_factor))} %")
        self.label_val_2.setText(f"{adjustments.rotation_deg} °")
        self.label_val_3.setText(str(adjustments.brightness))
        self.label_val_4.setText(str(adjustments.contrast))
        self.label_val_5.setText(str(adjustments.blur_radius))
        self.label_val_6.setText(f"{adjustments.opacity_pct} %")
        self.label_val_7.setText(str(adjustments.feather_radius))

    def _active_paste_effect_names(self) -> list[str]:
        names: list[str] = []
        if self._paste_shadow_enabled and self._paste_shadow_opacity_pct > 0 and self._paste_shadow_offset_px > 0:
            names.append("Shadow")
        if self._paste_motion_blur_enabled and self._paste_motion_blur_length > 1:
            names.append("Motion")
        return names

    def _refresh_paste_effects_button(self) -> None:
        active = self._active_paste_effect_names()
        self.btn_paste_effects.setText(f"Effects ({len(active)})")
        self.btn_paste_effects.setToolTip("")

    def _reset_paste_effects(self) -> None:
        self._paste_shadow_enabled = False
        self._paste_shadow_opacity_pct = 40
        self._paste_shadow_offset_px = 8
        self._paste_motion_blur_enabled = False
        self._paste_motion_blur_length = 9
        self._paste_motion_blur_angle_deg = 0
        self._refresh_paste_effects_button()

    def _edit_paste_effects(self) -> None:
        dialog = PasteEffectsDialog(
            self,
            shadow_enabled=self._paste_shadow_enabled,
            shadow_opacity_pct=self._paste_shadow_opacity_pct,
            shadow_offset_px=self._paste_shadow_offset_px,
            motion_blur_enabled=self._paste_motion_blur_enabled,
            motion_blur_length=self._paste_motion_blur_length,
            motion_blur_angle_deg=self._paste_motion_blur_angle_deg,
        )
        if dialog.exec() != QtWidgets.QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._paste_shadow_enabled = bool(values["shadow_enabled"])
        self._paste_shadow_opacity_pct = int(values["shadow_opacity_pct"])
        self._paste_shadow_offset_px = int(values["shadow_offset_px"])
        self._paste_motion_blur_enabled = bool(values["motion_blur_enabled"])
        self._paste_motion_blur_length = int(values["motion_blur_length"])
        self._paste_motion_blur_angle_deg = int(values["motion_blur_angle_deg"])
        self._refresh_paste_effects_button()
        if self.paste_candidate.has_anchor and self.paste_candidate.pasteimg is not None:
            self._paste_candidate_controller.recompute_preview()
        else:
            active = self._active_paste_effect_names()
            if active:
                self._set_paste_status_message(f"Advanced effects ready: {', '.join(active)}")
            else:
                self._refresh_paste_zone_status()

    def _prepare_paste_mode(self) -> None:
        self._clear_prediction_selection(redraw=False)
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        self.hideBox.setChecked(True)
        self.hideBbox(self.hideBox)

    def _smart_paste_enabled(self) -> bool:
        return self.combo_paste_mode.currentText() == 'Smart zone'

    def _set_paste_status_message(self, message: str) -> None:
        self.lbl_paste_status.setText(message)

    def _refresh_paste_zone_controls(self) -> None:
        has_image = getattr(self, "origin_canvas", None) is not None
        self.btn_set_zone.setDisabled(not has_image)
        self.btn_clear_zone.setDisabled(self.paste_candidate.smart_zone_rect is None)

    def _refresh_paste_zone_status(self) -> None:
        zone = self.paste_candidate.smart_zone_rect
        if self._smart_paste_enabled():
            if zone is None:
                self._set_paste_status_message("Smart zone mode: draw a valid placement region.")
                return
            x1, y1, x2, y2 = zone
            self._set_paste_status_message(
                f"Smart zone: {x2 - x1}x{y2 - y1} at ({x1}, {y1})"
            )
            return
        if zone is None:
            self._set_paste_status_message("Paste mode: manual")
            return
        x1, y1, x2, y2 = zone
        self._set_paste_status_message(
            f"Paste mode: manual (saved zone {x2 - x1}x{y2 - y1})"
        )

    def _current_paste_export_scale(self) -> float:
        canvas = self.canvas
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if canvas is None or ow <= 0 or oh <= 0 or canvas.width() <= 0 or canvas.height() <= 0:
            return 1.0
        return min(
            ow / float(canvas.width()),
            oh / float(canvas.height()),
        )

    @staticmethod
    def _format_paste_scale_hint_text(min_factor: float | None, max_factor: float | None) -> str:
        if min_factor is None and max_factor is None:
            return "n/a"
        if min_factor is None:
            return f"<= {int(round(max_factor * 100))}%"
        if max_factor is None:
            return f">= {int(round(min_factor * 100))}%"
        return f"{int(round(min_factor * 100))}-{int(round(max_factor * 100))}%"

    def _refresh_paste_size_hint(self) -> None:
        asset = self.paste_candidate.origin_pasteimg
        if asset is None:
            self.lbl_paste_size_hint.setText("Size hint: load asset")
            return
        asset_h, asset_w = asset.shape[:2]
        export_scale = self._current_paste_export_scale()
        target_size = self.combo_paste_size.currentText()
        hint = scale_hint_for_size_tag(
            asset_w,
            asset_h,
            target_size,
            export_scale=export_scale,
        )
        current_size = size_tag_for_scale_factor(
            asset_w,
            asset_h,
            self._get_paste_adjustments().scale_factor,
            export_scale=export_scale,
        )
        if not hint.reachable:
            self.lbl_paste_size_hint.setText(
                f"Size hint ({target_size}): unreachable at current zoom"
            )
            return
        self.lbl_paste_size_hint.setText(
            f"Size hint ({target_size}): "
            f"{self._format_paste_scale_hint_text(hint.min_factor, hint.max_factor)}"
            f" | Current: {current_size}"
        )

    def _on_paste_mode_changed(self, _index: int) -> None:
        if self.paste_candidate.has_anchor and self.paste_candidate.pasteimg is not None:
            self._paste_candidate_controller.recompute_preview()
            return
        self._refresh_paste_zone_status()

    def _begin_paste_zone_selection(self) -> None:
        if getattr(self, "origin_canvas", None) is None:
            return
        self._clear_prediction_selection(redraw=False)
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        self._paste_zone_drag_start = None
        self._image_canvas.set_mouse_move_handler(self._handle_paste_zone_move)
        self._image_canvas.set_mouse_press_handler(self._handle_paste_zone_press)
        self._image_canvas.set_mouse_release_handler(self._handle_paste_zone_release)
        self._image_canvas.set_mouse_leave_handler(self._handle_default_canvas_leave)
        self._set_paste_status_message("Drag on the image to define the smart zone.")

    def _clamp_canvas_point(self, x: int, y: int) -> tuple[int, int] | None:
        canvas = self.canvas
        if canvas is None or canvas.width() <= 0 or canvas.height() <= 0:
            return None
        return (
            min(max(int(x), 0), canvas.width() - 1),
            min(max(int(y), 0), canvas.height() - 1),
        )

    def _preview_paste_zone_selection(self, x1: int, y1: int, x2: int, y2: int) -> None:
        canvas = self.canvas
        if canvas is None:
            return
        preview = canvas.copy()
        draw_selection_overlay(
            preview,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            fill_color=QColor(45, 152, 218, 36),
            outline_color=QColor("#2D98DA"),
        )
        self._image_canvas.paint_label_only(preview)
        self.update()

    def _canvas_rect_to_origin_rect(
        self,
        x1: int,
        y1: int,
        x2: int,
        y2: int,
    ) -> tuple[int, int, int, int]:
        canvas = self.canvas
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if canvas is None or canvas.width() <= 0 or canvas.height() <= 0 or ow <= 0 or oh <= 0:
            return (0, 0, 0, 0)
        nx1, ny1, nx2, ny2 = normalize_rect(x1, y1, x2, y2)
        return normalize_rect(
            int(round(nx1 * ow / canvas.width())),
            int(round(ny1 * oh / canvas.height())),
            int(round(nx2 * ow / canvas.width())),
            int(round(ny2 * oh / canvas.height())),
        )

    def _handle_paste_zone_press(self, event) -> None:
        if event.button() != Qt.MouseButton.LeftButton:
            return
        point = self._clamp_canvas_point(int(event.position().x()), int(event.position().y()))
        if point is None:
            return
        self._paste_zone_drag_start = point
        self.__update_text_clicked_position(*point)
        self._preview_paste_zone_selection(*point, *point)

    def _handle_paste_zone_move(self, event) -> None:
        self.get_position(event)
        if self._paste_zone_drag_start is None:
            return
        point = self._clamp_canvas_point(int(event.position().x()), int(event.position().y()))
        if point is None:
            return
        self._preview_paste_zone_selection(
            self._paste_zone_drag_start[0],
            self._paste_zone_drag_start[1],
            point[0],
            point[1],
        )

    def _handle_paste_zone_release(self, event) -> None:
        start = self._paste_zone_drag_start
        self._paste_zone_drag_start = None
        self._install_default_canvas_handlers()
        if start is None:
            self._refresh_paste_zone_status()
            return
        point = self._clamp_canvas_point(int(event.position().x()), int(event.position().y()))
        if point is None:
            point = start
        if abs(point[0] - start[0]) < 4 or abs(point[1] - start[1]) < 4:
            self.set_img_ratio()
            return
        self.paste_candidate.smart_zone_rect = self._canvas_rect_to_origin_rect(
            start[0],
            start[1],
            point[0],
            point[1],
        )
        self.set_img_ratio()

    def _clear_paste_zone(self) -> None:
        if self.paste_candidate.smart_zone_rect is None:
            self._refresh_paste_zone_status()
            return
        self.paste_candidate.smart_zone_rect = None
        if getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()
        else:
            self._refresh_paste_zone_status()

    def controlpimg(self):
        self._paste_candidate_controller.recompute_preview()
        self._refresh_paste_size_hint()

    def pasteImg(self):
        self._paste_candidate_controller.enter_paste_mode()

    def newFile(self):
        filepath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self,
            directory=str(self._default_image_directory()),
            filter=self._image_file_filter(),
        )
        if filepath:
            if self._load_image_file(filepath, ask_confirm=True):
                self._sync_folder_images_from_path(filepath)

    def openFolder(self) -> None:
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Open image folder",
            str(self._default_image_directory()),
        )
        if not folder:
            return
        image_paths = list_supported_images(folder)
        if not image_paths:
            QtWidgets.QMessageBox.information(
                self,
                "Open image folder",
                "No supported images were found in this folder.",
            )
            self._set_folder_images([])
            return
        first_path = image_paths[0]
        if self._load_image_file(first_path, ask_confirm=True):
            self._set_folder_images(image_paths, current_path=first_path)

    def open_previous_image(self) -> None:
        self._load_adjacent_folder_image(-1)

    def open_next_image(self) -> None:
        self._load_adjacent_folder_image(1)

    def _confirm_open_image(self) -> bool:
        ret = self.mbox.question(
            self,
            'question',
            'New File?',
            self.mbox.StandardButton.Cancel,
            self.mbox.StandardButton.Ok,
        )
        return ret == self.mbox.StandardButton.Ok

    def _load_image_file(self, filepath: str, *, ask_confirm: bool) -> bool:
        if ask_confirm and not self._confirm_open_image():
            return False
        self._sync_current_prediction_review_state()

        img_data = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
        if img_data is None:
            QtWidgets.QMessageBox.critical(
                self,
                "Open image",
                "Failed to read this image file.",
            )
            return False

        self.imgfilePath = filepath
        self.setWindowTitle('ImgLab and ImgBlending   ' + filepath)
        self.img = img_data

        self.btn_zoom_in.setDisabled(False)
        self.slider_zoom.setDisabled(False)
        self.btn_zoom_out.setDisabled(False)
        self.btn_inputobj.setDisabled(False)
        self.btn_saveimg.setDisabled(False)
        self.btn_savelab.setDisabled(False)
        self.action_input.setDisabled(False)
        self.action_saveimg.setDisabled(False)
        self.action_savelab.setDisabled(False)

        self._paste_document.clear()
        self._paste_candidate.clear()
        self._gt_document.clear()
        self.predictions = []
        self._gt_draw.clear_pending()
        self._gt_edit.cancel_active_drag()
        self._annotation_controller.reset()
        self._install_default_canvas_handlers()

        self.gt_list_view.set_total(self._gt_document.total_boxes)
        self.pimg_list.setText(f'Paste Images  (Total: {self.paste_document.total_pastes})')
        self.gt_list_view.clear()
        self.pimglistwidget.clear()
        self._refresh_pred_listwidget()
        self._gt_workspace.clear_selection()
        self.resetVal()
        self.Hflip.setDisabled(True)
        self.Vflip.setDisabled(True)
        self._refresh_paste_zone_controls()
        self._refresh_paste_zone_status()

        if self.img.ndim == 2:
            img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
        elif self.img.ndim == 3 and self.img.shape[2] == 4:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGRA2RGB)
        elif self.img.ndim == 3:
            img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)
        else:
            QtWidgets.QMessageBox.critical(
                self,
                "Open image",
                "Unsupported image shape for display.",
            )
            return False

        self.origin_height, self.origin_width, self.origin_channel = img.shape
        bytesPerline = 3 * self.origin_width
        qimg = QImage(
            img, self.origin_width, self.origin_height,
            bytesPerline, QImage.Format.Format_RGB888
        )
        pm = QPixmap.fromImage(qimg)
        self.origin_canvas = pm.copy()
        self._image_canvas.set_canvas(pm)
        self.ratio_value = 50
        self._recompute_tile_grid()
        self.set_img_ratio()
        self._refresh_model_inference_ui()
        self._project_config.add_recent_image(filepath)
        self._auto_load_predictions_for_current_image()
        self._restart_autosave_timer()
        self._check_autosave_recovery(filepath)
        return True

    def inputObj(self):
        ClassMappingDialog(
            self,
            self,
            default_yaml_path=self._classes_yaml_path(),
        ).exec()

    def loadLabel(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self,
            directory=str(self._default_label_directory()),
            filter='Label (*.txt *.json);;TXT (*.txt);;JSON (*.json)',
        )
        if filePath:
            ret = self.mbox.question(
                self, 'question', 'New BoundingBox Label File?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                load_mode = self._prompt_load_label_mode()
                if load_mode is None:
                    return
                try:
                    mapping = class_mapping_from_object_list(self.object_list)
                    suffix = Path(filePath).suffix.lower()
                    if suffix == '.json':
                        annotations = import_json_label_file(
                            filePath,
                            class_mapping=mapping,
                            image_w=self.origin_width,
                            image_h=self.origin_height,
                            image_path=self.imgfilePath or None,
                        )
                    else:
                        annotations = import_yolo_hbb_label_file(
                            filePath,
                            class_mapping=mapping,
                            image_w=self.origin_width,
                            image_h=self.origin_height,
                        )
                    blocks = legacy_blocks_from_annotations(
                        annotations,
                        class_mapping=mapping,
                        canvas_w=self.origin_width,
                        canvas_h=self.origin_height,
                    )
                    if load_mode == "replace":
                        self._gt_actions.replace_with_blocks(blocks)
                    elif blocks:
                        self._gt_actions.append_blocks(blocks)
                    else:
                        self.gt_list_view.set_total(len(self.real_data))
                except (OSError, ValueError, IndexError):
                    self.mbox = QtWidgets.QMessageBox(self)
                    self.mbox.setText('This file is not in supported label format (YOLO txt / COCO json / metadata json)!')
                    self.mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                    self.mbox.exec()

    def _combined_annotation_records(self) -> list[dict]:
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        ow = getattr(self, "origin_width", None)
        oh = getattr(self, "origin_height", None)
        return build_combined_annotation_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            gt_real_data=self.real_data,
            gt_box_attributes=self.box_attributes,
            paste_real_data=self.real_pimg_data,
            object_list=self.object_list,
            class_id_to_super=sup,
        )

    def _combined_gt_boxes(self) -> list[tuple[str, float, float, float, float]]:
        return combined_gt_boxes(self.real_data, self.real_pimg_data)

    def _combined_box_attributes(self) -> list[dict[str, str]]:
        return combined_box_attributes(self.box_attributes, self.real_pimg_data)

    def _prompt_load_label_mode(self) -> str | None:
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setWindowTitle("Load Label")
        msg_box.setText("How should the loaded labels be applied?")
        append_btn = msg_box.addButton("Append GT", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        replace_btn = msg_box.addButton("Replace GT", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        msg_box.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        msg_box.setDefaultButton(append_btn)
        msg_box.exec()
        clicked = msg_box.clickedButton()
        if clicked == append_btn:
            return "append"
        if clicked == replace_btn:
            return "replace"
        return None

    def _prompt_statistics_scope(self) -> str | None:
        if not self.imgfilePath:
            QtWidgets.QMessageBox.information(
                self,
                "Statistics",
                "Open an image first so statistics can resolve the current image or folder.",
            )
            return None
        items = ["Current image", "Current folder"]
        if self._project_scope_available():
            items.append("Current project")
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Dataset Statistics",
            "Analysis scope:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current folder":
            return "folder"
        if choice == "Current project":
            return "project"
        return "image"

    def _prompt_error_analysis_scope(self) -> str | None:
        if not self.imgfilePath:
            QtWidgets.QMessageBox.information(
                self,
                "Error analysis",
                "Open an image first so error analysis can resolve the current image or folder.",
            )
            return None
        items = ["Current image", "Current folder"]
        if self._project_scope_available():
            items.append("Current project")
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Error Analysis",
            "Analysis scope:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current folder":
            return "folder"
        if choice == "Current project":
            return "project"
        return "image"

    def _prompt_fp_review_scope(self) -> str | None:
        if not self.imgfilePath:
            QtWidgets.QMessageBox.information(
                self,
                "FP-to-label review",
                "Open an image first so FP review can resolve the current folder or project.",
            )
            return None
        items = ["Current folder"]
        if self._project_scope_available():
            items.append("Current project")
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "FP-to-label review",
            "Build FP queue from:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current project":
            return "project"
        return "folder"

    def _prompt_validation_scope(self) -> str | None:
        if not self.imgfilePath:
            QtWidgets.QMessageBox.information(
                self,
                "Dataset QC",
                "Open an image first so dataset QC can resolve the current folder or project.",
            )
            return None
        items = ["Current folder"]
        if self._project_scope_available():
            items.append("Current project")
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Dataset QC",
            "Analysis scope:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current project":
            return "project"
        return "folder"

    def _prompt_review_summary_scope(self) -> str | None:
        if not self.imgfilePath:
            QtWidgets.QMessageBox.information(
                self,
                "Prediction review summary",
                "Open an image first so the review summary can resolve the current folder or project.",
            )
            return None
        items = ["Current folder"]
        if self._project_scope_available():
            items.append("Current project")
        choice, ok = QtWidgets.QInputDialog.getItem(
            self,
            "Prediction review summary",
            "Analysis scope:",
            items,
            0,
            False,
        )
        if not ok:
            return None
        if choice == "Current project":
            return "project"
        return "folder"

    def _current_folder_path(self) -> Path | None:
        if self._folder_image_paths:
            return Path(self._folder_image_paths[0]).parent
        if self.imgfilePath:
            return Path(self.imgfilePath).parent
        return None

    def _project_scope_available(self) -> bool:
        if self._project_config_path is None:
            return False
        try:
            return self._resolve_project_path(self._project_config.image_root).is_dir()
        except OSError:
            return False

    def _current_project_image_root(self) -> Path | None:
        if not self._project_scope_available():
            return None
        return self._resolve_project_path(self._project_config.image_root)

    def _folder_label_root_display(self) -> str:
        label_root = None
        if self._project_config_path is not None and self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        if label_root is not None and label_root.exists():
            return str(label_root)
        folder_path = self._current_folder_path()
        if folder_path is not None:
            inferred = folder_path.parent / "labels"
            if inferred.exists():
                return str(inferred)
        return "(same-folder sidecar / inferred)"

    def _project_label_root_display(self) -> str:
        if self._project_config_path is not None and self._project_config.label_root:
            return str(self._resolve_project_path(self._project_config.label_root))
        return "(project label root unavailable)"

    def _confirm_folder_error_analysis_run(self, result) -> bool:
        details = [
            f"Image folder: {result.folder_path}",
            f"Label root: {self._folder_label_root_display()}",
            f"Prediction folder: {result.prediction_root}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.labeled_images}",
            f"Predictions matched: {result.prediction_images}",
            f"Images analyzable: {result.analyzed_images}",
            f"Confidence threshold: {self._prediction_conf_threshold:.2f}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Error analysis summary",
            "Folder error analysis summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_folder_statistics_run(self, result) -> bool:
        details = [
            f"Image folder: {result.folder_path}",
            f"Label root: {self._folder_label_root_display()}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.labeled_images}",
            f"Annotations found: {len(result.records)}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Statistics summary",
            "Folder statistics summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_project_error_analysis_run(self, result) -> bool:
        details = [
            f"Image root: {result.folder_path}",
            f"Label root: {self._project_label_root_display()}",
            f"Prediction root: {result.prediction_root}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.labeled_images}",
            f"Predictions matched: {result.prediction_images}",
            f"Images analyzable: {result.analyzed_images}",
            f"Confidence threshold: {self._prediction_conf_threshold:.2f}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Project error analysis summary",
            "Project error analysis summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_project_statistics_run(self, result) -> bool:
        details = [
            f"Image root: {result.folder_path}",
            f"Label root: {self._project_label_root_display()}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.labeled_images}",
            f"Annotations found: {len(result.records)}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Project statistics summary",
            "Project statistics summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_folder_validation_run(self, result, prediction_root: Path | None) -> bool:
        details = [
            f"Image folder: {result.scope_path}",
            f"Label root: {self._folder_label_root_display()}",
            f"Prediction folder: {str(prediction_root) if prediction_root else '(not checked)'}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.matched_labels}",
            f"Predictions matched: {result.matched_predictions}",
            f"Issues found: {result.total_issues}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Dataset QC summary",
            "Folder dataset QC summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_project_validation_run(self, result, prediction_root: Path | None) -> bool:
        details = [
            f"Image root: {result.scope_path}",
            f"Label root: {self._project_label_root_display()}",
            f"Prediction root: {str(prediction_root) if prediction_root else '(not checked)'}",
            f"Images found: {result.total_images}",
            f"Labels matched: {result.matched_labels}",
            f"Predictions matched: {result.matched_predictions}",
            f"Issues found: {result.total_issues}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Project dataset QC summary",
            "Project dataset QC summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_folder_review_summary_run(self, report) -> bool:
        details = [
            f"Image folder: {report.scope_path}",
            f"Prediction folder: {report.prediction_root}",
            f"Images found: {report.total_images}",
            f"Images with predictions: {report.images_with_predictions}",
            f"Reviewed images: {report.reviewed_images}",
            f"Partial images: {report.partial_images}",
            f"Pending images: {report.pending_images}",
            f"No prediction images: {report.no_prediction_images}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Prediction review summary",
            "Folder prediction review summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _confirm_project_review_summary_run(self, report) -> bool:
        details = [
            f"Image root: {report.scope_path}",
            f"Prediction root: {report.prediction_root}",
            f"Images found: {report.total_images}",
            f"Images with predictions: {report.images_with_predictions}",
            f"Reviewed images: {report.reviewed_images}",
            f"Partial images: {report.partial_images}",
            f"Pending images: {report.pending_images}",
            f"No prediction images: {report.no_prediction_images}",
        ]
        answer = QtWidgets.QMessageBox.question(
            self,
            "Project prediction review summary",
            "Project prediction review summary:\n\n" + "\n".join(details) + "\n\nContinue?",
            QtWidgets.QMessageBox.StandardButton.Ok | QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok,
        )
        return answer == QtWidgets.QMessageBox.StandardButton.Ok

    def _prediction_review_root(self) -> Path:
        return self._project_config_base_dir()

    def _can_manage_prediction_review_session(self) -> bool:
        return self._prediction_folder_path is not None and self._current_folder_path() is not None

    def _prompt_prediction_review_resume_mode(self) -> str | None:
        if not self._can_manage_prediction_review_session():
            return "resume"
        if not has_prediction_review_session(
            image_folder=self._current_folder_path(),
            prediction_folder=self._prediction_folder_path,
            review_root=self._prediction_review_root(),
        ):
            return "resume"
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
        dialog.setWindowTitle("Prediction review")
        dialog.setText("Found saved review state for this image folder and prediction folder.")
        resume_button = dialog.addButton("Resume review", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        fresh_button = dialog.addButton("Start fresh", QtWidgets.QMessageBox.ButtonRole.DestructiveRole)
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == resume_button:
            return "resume"
        if clicked == fresh_button:
            return "fresh"
        return None

    def _load_saved_prediction_review_session(self) -> dict[str, PredictionReviewState]:
        if not self._can_manage_prediction_review_session():
            return {}
        states = load_prediction_review_session(
            image_folder=self._current_folder_path(),
            prediction_folder=self._prediction_folder_path,
            review_root=self._prediction_review_root(),
        )
        return states or {}

    def _save_prediction_review_session(self) -> None:
        if not self._can_manage_prediction_review_session():
            return
        save_prediction_review_session(
            image_folder=self._current_folder_path(),
            prediction_folder=self._prediction_folder_path,
            review_root=self._prediction_review_root(),
            states=self._prediction_review_states,
        )

    def _clear_saved_prediction_review_session(self) -> None:
        if not self._can_manage_prediction_review_session():
            return
        remove_prediction_review_session(
            image_folder=self._current_folder_path(),
            prediction_folder=self._prediction_folder_path,
            review_root=self._prediction_review_root(),
        )

    def clear_saved_prediction_review_state(self) -> None:
        if not self._can_manage_prediction_review_session():
            return
        answer = QtWidgets.QMessageBox.question(
            self,
            "Prediction review",
            "Clear saved review state for this image folder and prediction folder?",
            QtWidgets.QMessageBox.StandardButton.Yes | QtWidgets.QMessageBox.StandardButton.No,
            QtWidgets.QMessageBox.StandardButton.No,
        )
        if answer != QtWidgets.QMessageBox.StandardButton.Yes:
            return
        self._prediction_review_states.clear()
        self._clear_saved_prediction_review_session()
        self._auto_load_predictions_for_current_image()
        self._refresh_prediction_review_actions()

    def _resolve_project_path(self, value: str) -> Path:
        return resolve_project_path(
            self._project_config,
            value,
            config_path=self._project_config_path,
        )

    def _project_config_base_dir(self) -> Path:
        if self._project_config_path is not None:
            return self._project_config_path.parent
        return Path.cwd()

    def _classes_yaml_path(self) -> Path:
        return self._resolve_project_path(self._project_config.classes_yaml)

    def _default_image_directory(self) -> Path:
        current = Path(self.imgfilePath).parent if self.imgfilePath else None
        if current and current.is_dir():
            return current
        candidate = self._resolve_project_path(self._project_config.image_root)
        if candidate.is_dir():
            return candidate
        return self._project_config_base_dir()

    def _default_label_directory(self) -> Path:
        candidate = self._resolve_project_path(self._project_config.label_root)
        if candidate.is_dir():
            return candidate
        if self.imgfilePath:
            current = Path(self.imgfilePath).parent
            if current.is_dir():
                return current
        return self._project_config_base_dir()

    def _prediction_directory_for_current_image(self) -> Path | None:
        if not self.imgfilePath:
            return None
        current = Path(self.imgfilePath).parent
        parts = current.parts
        for idx in range(len(parts) - 1, -1, -1):
            if parts[idx].lower() != "images":
                continue
            candidate = Path(*parts[:idx], "predictions", *parts[idx + 1 :])
            if candidate.is_dir():
                return candidate
            break
        sibling = current.parent / "predictions"
        if sibling.is_dir():
            return sibling
        return None

    def _default_prediction_directory(self) -> Path:
        if self._prediction_folder_path is not None and self._prediction_folder_path.is_dir():
            return self._prediction_folder_path
        prediction_dir = self._prediction_directory_for_current_image()
        if prediction_dir is not None:
            return prediction_dir
        current = self._current_folder_path()
        if current is not None and current.is_dir():
            return current
        return self._default_label_directory()

    def _mapped_label_base_for_current_image(self) -> Path | None:
        if not self.imgfilePath:
            return None
        image_path = Path(self.imgfilePath)
        image_root = self._resolve_project_path(self._project_config.image_root)
        label_root = self._resolve_project_path(self._project_config.label_root)
        try:
            rel = image_path.resolve().relative_to(image_root.resolve())
        except ValueError:
            rel = None
        if rel is not None:
            return label_root / rel.parent / rel.stem
        return label_root / image_path.stem

    def _default_label_export_path(self, format_key: str) -> Path:
        base = self._mapped_label_base_for_current_image()
        if base is None:
            fallback_name = Path(self.imgfilePath).stem if self.imgfilePath else "labels"
            base = self._default_label_directory() / fallback_name
        suffix = {
            "yolo_hbb": ".txt",
            "bbox_txt": "_bbox.txt",
            "coco_json": ".json",
            "voc_xml": ".xml",
        }.get(format_key, ".txt")
        if suffix.startswith("_"):
            return base.parent / f"{base.name}{suffix}"
        return base.with_suffix(suffix)

    def _savelab_default_format_key(self) -> str:
        value = str(self._project_config.default_export_format or "yolo_hbb").strip().lower()
        if value in {"yolo", "yolo_hbb", "txt"}:
            return "yolo_hbb"
        if value in {"bbox", "bbox_txt", "bounding_boxes"}:
            return "bbox_txt"
        if value in {"coco", "coco_json"}:
            return "coco_json"
        if value in {"voc", "voc_xml", "pascal_voc_xml"}:
            return "voc_xml"
        return "yolo_hbb"

    def _prompt_prediction_folder(self) -> Path | None:
        start_dir = str(self._default_prediction_directory())
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "Select prediction folder",
            start_dir,
        )
        if not folder:
            return None
        return Path(folder)

    def _prompt_optional_validation_prediction_folder(self, scope: str) -> Path | bool | None:
        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Icon.Question)
        dialog.setWindowTitle("Dataset QC")
        dialog.setText(
            "Include prediction sidecar validation for the current "
            + ("project" if scope == "project" else "folder")
            + " scope?"
        )
        include_btn = dialog.addButton("Include predictions…", QtWidgets.QMessageBox.ButtonRole.AcceptRole)
        labels_only_btn = dialog.addButton("Labels only", QtWidgets.QMessageBox.ButtonRole.ActionRole)
        dialog.addButton(QtWidgets.QMessageBox.StandardButton.Cancel)
        dialog.exec()
        clicked = dialog.clickedButton()
        if clicked == include_btn:
            folder = self._prompt_prediction_folder()
            if folder is None:
                return False
            return folder
        if clicked == labels_only_btn:
            return None
        return False

    def _load_predictions_from_txt_path(self, path: Path) -> list:
        body = path.read_text(encoding="utf-8")
        return parse_predictions_yolo_txt(
            body,
            object_list=self.object_list,
            image_w=self.origin_width,
            image_h=self.origin_height,
        )

    def _review_key(self, image_path: str | Path) -> str:
        return str(Path(image_path).resolve())

    def _review_status_for_image(self, image_path: str | Path) -> str:
        state = self._prediction_review_states.get(self._review_key(image_path))
        if state is None:
            return "pending"
        return prediction_review_status(state)

    def _sync_current_prediction_review_state(
        self,
        *,
        accepted_delta: int = 0,
        rejected_delta: int = 0,
    ) -> None:
        if self._fp_review_queue:
            return
        if self._prediction_folder_path is None or not self.imgfilePath:
            return
        key = self._review_key(self.imgfilePath)
        state = self._prediction_review_states.get(key)
        if state is None:
            base_count = len(self.predictions) + int(accepted_delta) + int(rejected_delta)
            state = PredictionReviewState(
                original_count=max(0, base_count),
                remaining_predictions=tuple(clone_predictions(self.predictions)),
            )
        self._prediction_review_states[key] = update_prediction_review_state(
            state,
            accepted_delta=accepted_delta,
            rejected_delta=rejected_delta,
            remaining_predictions=self.predictions,
        )
        self._save_prediction_review_session()
        self._refresh_prediction_review_actions()

    def _auto_load_predictions_for_current_image(self) -> None:
        if self._prediction_folder_path is None or not self.imgfilePath or not getattr(self, "origin_width", None):
            self._refresh_prediction_review_actions()
            return
        key = self._review_key(self.imgfilePath)
        state = self._prediction_review_states.get(key)
        if state is not None:
            self.predictions = clone_predictions(state.remaining_predictions)
        else:
            try:
                self.predictions = load_prediction_sidecar(
                    self.imgfilePath,
                    prediction_root=self._prediction_folder_path,
                    object_list=self.object_list,
                    image_w=self.origin_width,
                    image_h=self.origin_height,
                )
                self._prediction_review_states[key] = initial_prediction_review_state(self.predictions)
                self._save_prediction_review_session()
            except (OSError, ValueError, IndexError):
                self.predictions = []
        self._refresh_pred_listwidget()
        if self.predictions:
            self.chk_show_preds.setChecked(True)
        if getattr(self, "origin_canvas", None) is not None:
            self.set_img_ratio()

    def _has_review_prediction_for_image(self, image_path: str | Path) -> bool:
        if self._prediction_folder_path is None:
            return False
        key = self._review_key(image_path)
        if key in self._prediction_review_states:
            return True
        return has_prediction_sidecar(image_path, prediction_root=self._prediction_folder_path)

    def _maybe_advance_prediction_review(self) -> None:
        if self._fp_review_queue:
            return
        if self._prediction_folder_path is None or not self.imgfilePath:
            return
        if self._review_status_for_image(self.imgfilePath) != "reviewed":
            return
        self._open_next_review_image(notify_if_missing=False)

    def open_next_review_image(self) -> None:
        self._open_next_review_image(notify_if_missing=True)

    def _open_next_review_image(self, *, notify_if_missing: bool) -> None:
        if self._prediction_folder_path is None or not self._folder_image_paths:
            if notify_if_missing:
                QtWidgets.QMessageBox.information(
                    self,
                    "Prediction review",
                    "Load a prediction folder and open an image folder first.",
                )
            return
        start_idx = self._folder_image_index
        for idx in range(start_idx + 1, len(self._folder_image_paths)):
            if not self._has_review_prediction_for_image(self._folder_image_paths[idx]):
                continue
            if self._review_status_for_image(self._folder_image_paths[idx]) == "reviewed":
                continue
            if self._load_image_file(self._folder_image_paths[idx], ask_confirm=False):
                self._folder_image_index = idx
                self._refresh_folder_navigation_ui()
            return
        if notify_if_missing:
            QtWidgets.QMessageBox.information(
                self,
                "Prediction review",
                "No later unreviewed images with prediction sidecars were found in the current folder.",
            )

    def _scan_current_folder_annotations(self):
        folder_path = self._current_folder_path()
        if folder_path is None:
            raise ValueError("No current folder available")
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        image_root = None
        label_root = None
        if self._project_config.image_root:
            image_root = self._resolve_project_path(self._project_config.image_root)
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        current_image_records = self._current_image_records_override()
        current_image_path = (self.imgfilePath or None) if current_image_records is not None else None
        return scan_folder_annotation_records(
            folder_path,
            object_list=self.object_list,
            class_id_to_super=sup,
            image_root=image_root,
            label_root=label_root,
            current_image_path=current_image_path,
            current_image_records=current_image_records,
        )

    def _scan_current_project_annotations(self):
        image_root = self._current_project_image_root()
        if image_root is None:
            raise ValueError("No current project image root available")
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        label_root = None
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        current_image_records = self._current_image_records_override()
        current_image_path = (self.imgfilePath or None) if current_image_records is not None else None
        return scan_folder_annotation_records(
            image_root,
            object_list=self.object_list,
            class_id_to_super=sup,
            recursive=True,
            image_root=image_root,
            label_root=label_root,
            current_image_path=current_image_path,
            current_image_records=current_image_records,
        )

    def _scan_current_folder_error_cases(self, prediction_root: Path):
        folder_path = self._current_folder_path()
        if folder_path is None:
            raise ValueError("No current folder available")
        image_root = None
        label_root = None
        if self._project_config.image_root:
            image_root = self._resolve_project_path(self._project_config.image_root)
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        current_gt_bundle = self._current_image_bundle_override()
        current_predictions = self._current_predictions_override(prediction_root)
        current_image_path = (self.imgfilePath or None) if current_gt_bundle is not None or current_predictions is not None else None
        return scan_folder_error_cases(
            folder_path,
            object_list=self.object_list,
            prediction_root=prediction_root,
            image_root=image_root,
            label_root=label_root,
            current_image_path=current_image_path,
            current_image_gt_bundle=current_gt_bundle,
            current_image_predictions=current_predictions,
            min_confidence=self._prediction_conf_threshold,
        )

    def _scan_current_project_error_cases(self, prediction_root: Path):
        image_root = self._current_project_image_root()
        if image_root is None:
            raise ValueError("No current project image root available")
        label_root = None
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        current_gt_bundle = self._current_image_bundle_override()
        current_predictions = self._current_predictions_override(prediction_root)
        current_image_path = (self.imgfilePath or None) if current_gt_bundle is not None or current_predictions is not None else None
        return scan_folder_error_cases(
            image_root,
            object_list=self.object_list,
            prediction_root=prediction_root,
            recursive=True,
            image_root=image_root,
            label_root=label_root,
            current_image_path=current_image_path,
            current_image_gt_bundle=current_gt_bundle,
            current_image_predictions=current_predictions,
            min_confidence=self._prediction_conf_threshold,
        )

    def _scan_current_folder_validation(self, prediction_root: Path | None):
        folder_path = self._current_folder_path()
        if folder_path is None:
            raise ValueError("No current folder available")
        image_root = None
        label_root = None
        if self._project_config.image_root:
            image_root = self._resolve_project_path(self._project_config.image_root)
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        return scan_dataset_validation(
            folder_path,
            object_list=self.object_list,
            image_root=image_root,
            label_root=label_root,
            prediction_root=prediction_root,
        )

    def _scan_current_project_validation(self, prediction_root: Path | None):
        image_root = self._current_project_image_root()
        if image_root is None:
            raise ValueError("No current project image root available")
        label_root = None
        if self._project_config.label_root:
            label_root = self._resolve_project_path(self._project_config.label_root)
        return scan_dataset_validation(
            image_root,
            object_list=self.object_list,
            recursive=True,
            image_root=image_root,
            label_root=label_root,
            prediction_root=prediction_root,
        )

    def _scan_current_folder_review_report(self, prediction_root: Path):
        folder_path = self._current_folder_path()
        if folder_path is None:
            raise ValueError("No current folder available")
        image_root = None
        if self._project_config.image_root:
            image_root = self._resolve_project_path(self._project_config.image_root)
        return scan_prediction_review_report(
            folder_path,
            prediction_root=prediction_root,
            review_root=self._prediction_review_root(),
            image_root=image_root,
            current_states=self._prediction_review_states,
        )

    def _scan_current_project_review_report(self, prediction_root: Path):
        image_root = self._current_project_image_root()
        if image_root is None:
            raise ValueError("No current project image root available")
        return scan_prediction_review_report(
            image_root,
            prediction_root=prediction_root,
            review_root=self._prediction_review_root(),
            recursive=True,
            image_root=image_root,
            current_states=self._prediction_review_states,
        )

    def _has_current_annotation_override(self) -> bool:
        return bool(
            getattr(self, "real_data", None)
            or getattr(self, "real_pimg_data", None)
            or self._annotation_controller.can_undo()
            or self._annotation_controller.can_redo()
        )

    def _current_image_records_override(self):
        if not self._has_current_annotation_override():
            return None
        return self._combined_annotation_records()

    def _current_image_bundle_override(self):
        if not self._has_current_annotation_override():
            return None
        return self._current_image_annotation_bundle()

    def _prediction_root_matches_loaded(self, prediction_root: Path) -> bool:
        if self._prediction_folder_path is None:
            return False
        try:
            return Path(prediction_root).resolve() == self._prediction_folder_path.resolve()
        except OSError:
            return False

    def _has_current_prediction_override(self, prediction_root: Path) -> bool:
        if self.predictions:
            return True
        if not self.imgfilePath or not self._prediction_root_matches_loaded(prediction_root):
            return False
        return self._review_key(self.imgfilePath) in self._prediction_review_states

    def _current_predictions_override(self, prediction_root: Path):
        if not self._has_current_prediction_override(prediction_root):
            return None
        return self.predictions

    def _current_image_annotation_bundle(self):
        records = self._combined_annotation_records()
        gt_boxes = self._combined_gt_boxes()
        gt_attributes = self._combined_box_attributes()
        ow = getattr(self, "origin_width", 0) or 0
        oh = getattr(self, "origin_height", 0) or 0
        from sdde.dataset_scan import ImageAnnotationBundle

        return ImageAnnotationBundle(
            image_path=self.imgfilePath or "",
            image_width=ow,
            image_height=oh,
            records=tuple(dict(rec) for rec in records),
            gt_boxes=tuple(gt_boxes),
            gt_attributes=tuple(dict(attrs) for attrs in gt_attributes),
            has_label=bool(records),
        )

    def showLabel(self):
        showimg = self.build_export_image_pixmap()
        if showimg is None:
            return
        self.nw4 = ShowlabWindow(main_widget=self, showimg=showimg)
        self.nw4.show()

    def saveFile(self):
        self.nw = SaveimgWindow(main_widget=self)
        self.nw.show()

    def saveLabel(self):
        self.nw3 = SavelabWindow(main_widget=self)
        self.nw3.show()

    def closeFile(self):
        self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.is_confirm_quit:
            reply = self.mbox.question(
                self, 'question', 'Quit Application?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if reply == self.mbox.StandardButton.Ok:
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
