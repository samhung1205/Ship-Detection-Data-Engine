"""
Main application window for ImgLab and ImgBlending.
"""
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtGui import (
    QAction, QPixmap, QImage, QColor, QCloseEvent,
)
from PyQt6.QtCore import Qt, QRect, QTimer
import cv2

from .constants import (
    STYLE_BUTTON_PRIMARY,
    STYLE_BUTTON_SECONDARY,
    STYLE_BUTTON_SECONDARY_DISABLED,
    STYLE_LIST_WIDGET,
)
from .canvas_utils import draw_paste_images_on_canvas
from .annotation_actions_controller import AnnotationActionsController
from .annotation_draw_controller import AnnotationDrawController
from .annotation_edit_controller import AnnotationEditController
from .annotation_list_controller import AnnotationListController
from .annotation_preview_controller import AnnotationPreviewController
from .annotation_list_view import AnnotationListView
from .paste_candidate_controller import PasteCandidateController
from .paste_actions_controller import PasteActionsController
from .paste_preview_controller import PastePreviewController
from .annotation_workspace_controller import AnnotationWorkspaceController
from .canvas_widget import ImageCanvasWidget
from .annotation_controller import AnnotationController
from sdde.class_catalog import ClassCatalog

from .class_mapping_service import load_class_catalog
from .attribute_panel import AttributePanel
from .dialogs import ClassMappingDialog, ErrorAnalysisDialog, StatisticsDialog, ShowlabWindow, SaveimgWindow, SavelabWindow
from .tile_panel import TilePanel
from sdde.tile import TileConfig, TileRect, compute_tile_grid
from sdde.metadata_export import export_annotations_csv, export_annotations_json
from sdde.prediction import (
    STATUS_EDITED,
    STATUS_PREDICTED,
    parse_predictions_yolo_txt,
)
from sdde.augmentation import (
    PasteRecord,
    bbox_from_legacy_paste_row,
    export_paste_records_csv,
    export_paste_records_json,
)
from sdde.project_config import ProjectConfig, load_project_config, save_project_config
from sdde.autosave import has_autosave, read_autosave, remove_autosave, write_autosave
from sdde.document import AnnotationDocument
from sdde.paste_candidate import PasteCandidateSession
from sdde.paste_document import PasteDocument
from sdde.import_export import import_yolo_hbb_label_file
from sdde.legacy_rows import class_mapping_from_object_list, legacy_blocks_from_annotations


class MyWidget(QtWidgets.QWidget):
    def __init__(self, is_confirm_quit: bool = True):
        super().__init__()
        self.setWindowTitle('Ship Detection Data Engine')
        self.resize(1460, 760)
        self.setUpdatesEnabled(True)
        self.is_confirm_quit = is_confirm_quit
        self.object_list = []
        self._gt_document = AnnotationDocument()
        self._paste_candidate = PasteCandidateSession()
        self._paste_document = PasteDocument()
        self.imgfilePath = ''
        self.predictions: list = []
        self._tile_grid: list[TileRect] = []
        self._project_config = ProjectConfig()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self._annotation_controller = AnnotationController(self)
        self.ui()
        self.adjustUi()
        self._bootstrap_class_catalog()
        self._try_load_default_project_config()

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
        self.class_catalog = load_class_catalog()
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
        self.btn_loadpred.setDisabled(False)
        self.action_load_pred.setDisabled(False)
        self.action_clear_pred.setDisabled(False)
        self.pred_listwidget.setDisabled(False)
        self.btn_pred_accept.setDisabled(False)
        self.btn_pred_reject.setDisabled(False)

    def _try_load_default_project_config(self) -> None:
        """Load project_config.yaml from CWD if it exists."""
        cfg_path = Path("project_config.yaml")
        if cfg_path.exists():
            try:
                self._project_config = load_project_config(cfg_path)
                self._apply_project_config()
            except (OSError, ValueError):
                pass

    def _apply_project_config(self) -> None:
        """Push ProjectConfig values into UI widgets."""
        cfg = self._project_config
        self._tile_panel.spin_size.setValue(cfg.tile_size)
        self._tile_panel.spin_stride.setValue(cfg.tile_stride)
        self._restart_autosave_timer()

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
            self, "Open project config", "", "YAML (*.yaml *.yml)"
        )
        if not fp:
            return
        try:
            self._project_config = load_project_config(fp)
            self._apply_project_config()
        except (OSError, ValueError) as e:
            QtWidgets.QMessageBox.critical(self, "Load failed", str(e))

    def save_project_config_as(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save project config", "project_config.yaml", "YAML (*.yaml *.yml)"
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
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Save failed", str(e))

    def export_annotation_metadata_json(self) -> None:
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        ow = getattr(self, "origin_width", None)
        oh = getattr(self, "origin_height", None)
        recs = self.gt_document.build_metadata_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            object_list=self.object_list,
            class_id_to_super=sup,
        )
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
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        ow = getattr(self, "origin_width", None)
        oh = getattr(self, "origin_height", None)
        recs = self.gt_document.build_metadata_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            object_list=self.object_list,
            class_id_to_super=sup,
        )
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
        ### 畫布（捲動、縮放重繪、bbox / paste 疊圖；滑鼠事件由 hook 轉回 MyWidget）###
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(125, 30, 870, 480)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)

        self._image_canvas = ImageCanvasWidget(self.verticalLayoutWidget)
        self.verticalLayout.addWidget(self._image_canvas)
        self.pmap = self._image_canvas.image_label
        self.pmap.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

        ### control zoom ###
        self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_in.setGeometry(QRect(155, 520, 89, 25))
        self.btn_zoom_in.setText("zoom_in")
        self.btn_zoom_in.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_in.setDisabled(True)
        self.btn_zoom_in.clicked.connect(self.set_zoom_in)

        self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.slider_zoom.setGeometry(QRect(255, 518, 231, 28))
        self.slider_zoom.setProperty("value", 50)
        self.slider_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.slider_zoom.setDisabled(True)
        self.slider_zoom.valueChanged.connect(self.getslidervalue)

        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QRect(615, 520, 300, 24))

        self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_out.setGeometry(QRect(505, 520, 89, 25))
        self.btn_zoom_out.setText("zoom_out")
        self.btn_zoom_out.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_out.setDisabled(True)
        self.btn_zoom_out.clicked.connect(self.set_zoom_out)

        ### mouseMove（由 ImageCanvasWidget 轉發）###
        self.label_get_pos = QtWidgets.QLabel(self)
        self.label_get_pos.setGeometry(125, 552, 250, 18)
        self.label_get_pos.setText('current position = (x,y)')
        self.label_get_pos.setStyleSheet('font-size: 12px;')

        ### mousePress（由 ImageCanvasWidget 轉發；標註模式會改為 paint / paste）###
        self.label_click_pos = QtWidgets.QLabel(self)
        self.label_click_pos.setGeometry(395, 552, 260, 18)
        self.label_click_pos.setText('clicked position = (x,y)')
        self.label_click_pos.setStyleSheet('font-size: 12px;')

        ### show img.shape ###
        self.label_img_shape = QtWidgets.QLabel(self)
        self.label_img_shape.setGeometry(125, 574, 870, 18)

        self.lbl_autosave_status = QtWidgets.QLabel(self)
        self.lbl_autosave_status.setGeometry(125, 596, 300, 16)
        self.lbl_autosave_status.setStyleSheet("font-size: 11px; color: #888;")
        self.lbl_autosave_status.setText("")

        ### label_button ###
        self.btn_label = QtWidgets.QPushButton(self)
        self.btn_label.setText('Create RectBox')
        self.btn_label.setGeometry(1010, 30, 110, 24)
        self.btn_label.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_label.setDisabled(True)
        self.btn_label.clicked.connect(self.make_label)

        ### label_list ###
        self.label_list = QtWidgets.QLabel(self)
        self.label_list.setText('Box Labels')
        self.label_list.setGeometry(1010, 56, 200, 20)
        self.label_list.setStyleSheet('font-size: 12px;')

        self.hideBox = QtWidgets.QCheckBox(self)
        self.hideBox.move(1010, 76)
        self.hideBox.setText('Hide Box')
        self.hideBox.clicked.connect(lambda: self.hideBbox(self.hideBox))

        self.chk_show_preds = QtWidgets.QCheckBox(self)
        self.chk_show_preds.move(1110, 76)
        self.chk_show_preds.setText('Show preds')
        self.chk_show_preds.setChecked(True)
        self.chk_show_preds.clicked.connect(self.set_img_ratio)

        self.listwidget = QtWidgets.QListWidget(self)
        self.listwidget.addItems([])
        self.listwidget.setGeometry(1010, 96, 430, 110)
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

        self._attr_panel = AttributePanel(self)
        self._attr_panel.setGeometry(1010, 212, 430, 170)
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
        self.label_clear.setGeometry(1380, 386, 60, 20)
        self.label_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.label_clear.clicked.connect(self._gt_list_controller.confirm_clear_all)

        ### paste_button ###
        self.btn_paste = QtWidgets.QPushButton(self)
        self.btn_paste.setText('Paste Image')
        self.btn_paste.setGeometry(1010, 410, 110, 24)
        self.btn_paste.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_paste.setDisabled(True)
        self.btn_paste.clicked.connect(self.pasteImg)

        self.label_pasteimg = QtWidgets.QLabel(self)
        self.label_pasteimg.setText('Image')
        self.label_pasteimg.setGeometry(1010, 436, 80, 20)
        self.label_pasteimg.setStyleSheet('font-size: 12px;')

        self.Hflip = QtWidgets.QCheckBox(self)
        self.Hflip.move(1060, 436)
        self.Hflip.setText('HorizontalFlip')
        self.Hflip.setDisabled(True)
        self.Hflip.clicked.connect(lambda: self.Hflippimg(self.Hflip))

        self.white_canvas = QPixmap(80, 80)
        self.white_canvas.fill(QColor('#ffffff'))
        self.pmap_pasteimg = QtWidgets.QLabel(self)
        self.pmap_pasteimg.setGeometry(1010, 456, 80, 80)
        self.pmap_pasteimg.setStyleSheet('border: 1px solid #D3D3D3;')
        self.pmap_pasteimg.setPixmap(self.white_canvas)

        self.btn_chooseimg = QtWidgets.QPushButton(self)
        self.btn_chooseimg.setText('Choose')
        self.btn_chooseimg.setGeometry(1010, 558, 50, 20)
        self.btn_chooseimg.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_chooseimg.clicked.connect(self.chooseImg)

        self.btn_add = QtWidgets.QPushButton(self)
        self.btn_add.setText('Add')
        self.btn_add.setGeometry(1065, 558, 50, 20)
        self.btn_add.setStyleSheet(STYLE_BUTTON_SECONDARY_DISABLED)
        self.btn_add.setDisabled(True)
        self.btn_add.clicked.connect(self.inputPimg)

        self.btn_reset = QtWidgets.QPushButton(self)
        self.btn_reset.setText('Reset')
        self.btn_reset.setGeometry(1390, 558, 50, 20)
        self.btn_reset.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_reset.clicked.connect(self.resetVal)

        ### Paste_imgae_QListWidget ###
        self.pimg_list = QtWidgets.QLabel(self)
        self.pimg_list.setText('Paste Images')
        self.pimg_list.setGeometry(1010, 584, 200, 20)
        self.pimg_list.setStyleSheet('font-size: 12px;')

        self.pimglistwidget = QtWidgets.QListWidget(self)
        self.pimglistwidget.addItems([])
        self.pimglistwidget.setGeometry(1010, 606, 430, 74)
        self.pimglistwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.pimglistwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)

        self.pimg_clear = QtWidgets.QPushButton(self)
        self.pimg_clear.setText('Delete all')
        self.pimg_clear.setGeometry(1380, 684, 60, 20)
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
            on_prepare_paste_mode=self._prepare_paste_mode,
            on_clicked_position=self.__update_text_clicked_position,
            on_enable_add=lambda enabled: self.btn_add.setDisabled(not enabled),
            on_set_adjustment_labels=self._set_paste_adjustment_labels,
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
        self.action_save_project.triggered.connect(self.save_project_config_as)
        self.menu_file.addAction(self.action_save_project)
        self.menu_file.addSeparator()

        self.action_open = QAction('Open Image')
        self.action_open.setShortcut('Ctrl+o')
        self.action_open.triggered.connect(self.newFile)
        self.menu_file.addAction(self.action_open)

        self.action_input = QAction('Class mapping')
        self.action_input.setShortcut('Ctrl+i')
        self.action_input.setDisabled(True)
        self.action_input.triggered.connect(self.inputObj)
        self.menu_file.addAction(self.action_input)
        self.menu_file.addSeparator()

        self.action_load = QAction('Load Label')
        self.action_load.setShortcut('Ctrl+l')
        self.action_load.setDisabled(True)
        self.action_load.triggered.connect(self.loadLabel)
        self.menu_file.addAction(self.action_load)

        self.action_load_pred = QAction('Load predictions…')
        self.action_load_pred.setDisabled(True)
        self.action_load_pred.triggered.connect(self.load_predictions)
        self.menu_file.addAction(self.action_load_pred)
        self.action_clear_pred = QAction('Clear predictions')
        self.action_clear_pred.setDisabled(True)
        self.action_clear_pred.triggered.connect(self.clear_predictions)
        self.menu_file.addAction(self.action_clear_pred)
        self.menu_file.addSeparator()

        self.action_saveimg = QAction('Save Image')
        self.action_saveimg.setDisabled(True)
        self.action_saveimg.triggered.connect(self.saveFile)
        self.menu_file.addAction(self.action_saveimg)

        self.action_savelab = QAction('Save Label')
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
        self.action_label.setDisabled(True)
        self.action_label.triggered.connect(self.make_label)
        self.menu_edit.addAction(self.action_label)

        self.action_paste = QAction('Paste Image')
        self.action_paste.setDisabled(True)
        self.action_paste.triggered.connect(self.pasteImg)
        self.menu_edit.addAction(self.action_paste)
        self.menu_edit.addSeparator()

        self.action_show = QAction('Show Label')
        self.action_show.setDisabled(True)
        self.action_show.triggered.connect(self.showLabel)
        self.menu_edit.addAction(self.action_show)
        self.menu_edit.addSeparator()

        self.action_undo = QAction('Undo')
        self.action_undo.setShortcut('Ctrl+Z')
        self.action_undo.triggered.connect(self._on_annotation_undo)
        self.menu_edit.addAction(self.action_undo)

        self.action_redo = QAction('Redo')
        self.action_redo.setShortcut('Ctrl+Shift+Z')
        self.action_redo.triggered.connect(self._on_annotation_redo)
        self.menu_edit.addAction(self.action_redo)

        self.menubar.addMenu(self.menu_edit)

        ### menu_Analysis ###
        self.menu_analysis = QtWidgets.QMenu('Analysis')
        self.action_run_error = QAction('Run error analysis…')
        self.action_run_error.triggered.connect(self.run_error_analysis)
        self.menu_analysis.addAction(self.action_run_error)
        self.action_show_stats = QAction('Dataset statistics…')
        self.action_show_stats.triggered.connect(self.show_statistics)
        self.menu_analysis.addAction(self.action_show_stats)
        self.menubar.addMenu(self.menu_analysis)

        ### open_button ###
        self.btn_open = QtWidgets.QPushButton(self)
        self.btn_open.setText('Open Image')
        self.btn_open.setGeometry(10, 30, 105, 24)
        self.btn_open.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_open.clicked.connect(self.newFile)

        self.btn_inputobj = QtWidgets.QPushButton(self)
        self.btn_inputobj.setText('Class mapping')
        self.btn_inputobj.setGeometry(10, 58, 105, 24)
        self.btn_inputobj.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_inputobj.setDisabled(True)
        self.btn_inputobj.clicked.connect(self.inputObj)

        self.btn_loadlab = QtWidgets.QPushButton(self)
        self.btn_loadlab.setText('Load Label')
        self.btn_loadlab.setGeometry(10, 90, 105, 24)
        self.btn_loadlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_loadlab.setDisabled(True)
        self.btn_loadlab.clicked.connect(self.loadLabel)

        self.btn_showlab = QtWidgets.QPushButton(self)
        self.btn_showlab.setText('Show Label')
        self.btn_showlab.setGeometry(10, 118, 105, 24)
        self.btn_showlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_showlab.setDisabled(True)
        self.btn_showlab.clicked.connect(self.showLabel)

        self.btn_loadpred = QtWidgets.QPushButton(self)
        self.btn_loadpred.setText('Load preds')
        self.btn_loadpred.setGeometry(10, 150, 105, 24)
        self.btn_loadpred.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_loadpred.setDisabled(True)
        self.btn_loadpred.clicked.connect(self.load_predictions)

        self.pred_listwidget = QtWidgets.QListWidget(self)
        self.pred_listwidget.setGeometry(10, 178, 105, 110)
        self.pred_listwidget.setStyleSheet('QListWidget::item{font-size:11px;}')
        self.pred_listwidget.setDisabled(True)

        self.btn_pred_accept = QtWidgets.QPushButton(self)
        self.btn_pred_accept.setText('Accept')
        self.btn_pred_accept.setGeometry(10, 292, 50, 22)
        self.btn_pred_accept.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_pred_accept.setDisabled(True)
        self.btn_pred_accept.clicked.connect(self.accept_selected_prediction)

        self.btn_pred_reject = QtWidgets.QPushButton(self)
        self.btn_pred_reject.setText('Reject')
        self.btn_pred_reject.setGeometry(64, 292, 50, 22)
        self.btn_pred_reject.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_pred_reject.setDisabled(True)
        self.btn_pred_reject.clicked.connect(self.reject_selected_prediction)

        self._tile_panel = TilePanel(self)
        self._tile_panel.setGeometry(5, 322, 118, 170)
        self._tile_panel.tile_config_changed.connect(self._on_tile_config_changed)
        self._tile_panel.tile_index_changed.connect(self._on_tile_index_changed)
        self._tile_panel.tile_view_toggled.connect(self._on_tile_view_toggled)

        self.btn_saveimg = QtWidgets.QPushButton(self)
        self.btn_saveimg.setText('Save Image')
        self.btn_saveimg.setGeometry(10, 504, 105, 24)
        self.btn_saveimg.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_saveimg.setDisabled(True)
        self.btn_saveimg.clicked.connect(self.saveFile)

        self.btn_savelab = QtWidgets.QPushButton(self)
        self.btn_savelab.setText('Save Label')
        self.btn_savelab.setGeometry(10, 532, 105, 24)
        self.btn_savelab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_savelab.setDisabled(True)
        self.btn_savelab.clicked.connect(self.saveLabel)

        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('Quit')
        self.btn_close.setGeometry(10, 564, 105, 24)
        self.btn_close.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_close.clicked.connect(self.closeFile)

    def adjustUi(self):
        sx = 1130
        sw = 180

        self.label_adj_1 = QtWidgets.QLabel(self)
        self.label_adj_1.setGeometry(sx, 456, 70, 14)
        self.label_adj_1.setText('Resize')
        self.label_adj_1.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_1 = QtWidgets.QSlider(self)
        self.slider_1.setOrientation(Qt.Orientation.Horizontal)
        self.slider_1.setGeometry(sx + 80, 451, sw, 28)
        self.slider_1.setRange(0, 100)
        self.slider_1.setValue(50)
        self.slider_1.valueChanged.connect(self.controlpimg)
        self.label_val_1 = QtWidgets.QLabel(self)
        self.label_val_1.setGeometry(sx + 265, 451, 50, 28)
        self.label_val_1.setText("100 %")
        self.label_val_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_2 = QtWidgets.QLabel(self)
        self.label_adj_2.setGeometry(sx, 480, 70, 14)
        self.label_adj_2.setText('Rotate')
        self.label_adj_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_2 = QtWidgets.QSlider(self)
        self.slider_2.setOrientation(Qt.Orientation.Horizontal)
        self.slider_2.setGeometry(sx + 80, 475, sw, 28)
        self.slider_2.setRange(0, 360)
        self.slider_2.setValue(0)
        self.slider_2.valueChanged.connect(self.controlpimg)
        self.label_val_2 = QtWidgets.QLabel(self)
        self.label_val_2.setGeometry(sx + 265, 475, 50, 28)
        self.label_val_2.setText('0 °')
        self.label_val_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_3 = QtWidgets.QLabel(self)
        self.label_adj_3.setGeometry(sx, 504, 70, 14)
        self.label_adj_3.setText('Brightness')
        self.label_adj_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_3 = QtWidgets.QSlider(self)
        self.slider_3.setOrientation(Qt.Orientation.Horizontal)
        self.slider_3.setGeometry(sx + 80, 499, sw, 28)
        self.slider_3.setRange(0, 200)
        self.slider_3.setValue(100)
        self.slider_3.valueChanged.connect(self.controlpimg)
        self.label_val_3 = QtWidgets.QLabel(self)
        self.label_val_3.setGeometry(sx + 265, 499, 50, 28)
        self.label_val_3.setText('100')
        self.label_val_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_4 = QtWidgets.QLabel(self)
        self.label_adj_4.setGeometry(sx, 528, 70, 14)
        self.label_adj_4.setText('Contrast')
        self.label_adj_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_4 = QtWidgets.QSlider(self)
        self.slider_4.setOrientation(Qt.Orientation.Horizontal)
        self.slider_4.setGeometry(sx + 80, 523, sw, 28)
        self.slider_4.setRange(0, 200)
        self.slider_4.setValue(100)
        self.slider_4.valueChanged.connect(self.controlpimg)
        self.label_val_4 = QtWidgets.QLabel(self)
        self.label_val_4.setGeometry(sx + 265, 523, 50, 28)
        self.label_val_4.setText('100')
        self.label_val_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def getslidervalue(self):
        self.set_slider_value(self.slider_zoom.value() + 1)

    def _current_tile_rect(self) -> tuple[int, int, int, int] | None:
        if not self._tile_panel.is_enabled() or not self._tile_grid:
            return None
        idx = self._tile_panel.current_index()
        if idx < 0 or idx >= len(self._tile_grid):
            return None
        t = self._tile_grid[idx]
        return (t.x, t.y, t.w, t.h)

    def set_img_ratio(self, *, bbox_data_override: list | None = None):
        ow = int(getattr(self, "origin_width", 0) or 0)
        self.ratio_rate, self.qpixmap_height = self._image_canvas.redraw_scaled_overlay(
            origin_canvas=self.origin_canvas,
            ratio_value=self.ratio_value,
            origin_height=self.origin_height,
            origin_width=ow,
            hide_boxes=self.hideBox.isChecked(),
            bbox_data=self.data if bbox_data_override is None else bbox_data_override,
            pimg_data=self.pimg_data,
            paste_images=self.paste_images,
            predictions=self.predictions,
            show_predictions=self.chk_show_preds.isChecked(),
            tile_rect=self._current_tile_rect(),
        )
        self.update()
        self.__update_text_ratio()
        self.__update_text_img_shape()

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
        except (AttributeError, RuntimeError):
            self.__update_text_get_position(mx, my)

    def __update_text_clicked_position(self, x, y):
        self.label_click_pos.setText(f'Clicked position = ({x}, {y})')

    def _install_default_canvas_handlers(self) -> None:
        self._image_canvas.set_mouse_move_handler(self._handle_default_canvas_move)
        self._image_canvas.set_mouse_press_handler(self._handle_default_canvas_press)
        self._image_canvas.set_mouse_release_handler(self._handle_default_canvas_release)

    def _handle_default_canvas_move(self, event) -> None:
        self.get_position(event)
        self._gt_edit.handle_move(event)

    def _handle_default_canvas_press(self, event) -> None:
        self.get_clicked_position(event)
        self._gt_edit.handle_press(event)

    def _handle_default_canvas_release(self, event) -> None:
        self._gt_edit.handle_release(event)

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

    def _on_gt_row_changed(self, row: int) -> None:
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        self._install_default_canvas_handlers()
        self._gt_list_controller.on_row_changed(row)

    def get_clicked_position(self, event):
        mx = int(event.position().x())
        my = int(event.position().y())
        self.__update_text_clicked_position(mx, my)

    def __update_text_img_shape(self):
        current_text = f"Current img shape = ({self.canvas.width()}, {self.canvas.height()})"
        origin_text = f"Origin img shape = ({self.origin_width}, {self.origin_height})"
        self.label_img_shape.setText(current_text + "    |    " + origin_text)

    def make_label(self):
        self._gt_edit.cancel_active_drag()
        self._gt_draw.enter_draw_mode()

    def _prepare_gt_draw_mode(self) -> None:
        self.hideBox.setChecked(False)
        self.hideBbox(self.hideBox)

    def hideBbox(self, _cb) -> None:
        try:
            self.set_img_ratio()
        except (AttributeError, ZeroDivisionError):
            return

    def _refresh_pred_listwidget(self) -> None:
        self.pred_listwidget.clear()
        for p in self.predictions:
            st = getattr(p, "pred_status", "")
            conf = float(getattr(p, "confidence", 0.0))
            name = getattr(p, "class_name", "")
            suffix = "" if st in (STATUS_PREDICTED, STATUS_EDITED) else f" [{st}]"
            self.pred_listwidget.addItem(f"{name} {conf:.2f}{suffix}")

    def load_predictions(self) -> None:
        if not getattr(self, "origin_width", None) or not self.object_list:
            QtWidgets.QMessageBox.information(
                self, "Load predictions", "Open an image first, and ensure class mapping is loaded."
            )
            return
        fp, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, directory="dataset", filter="TXT (*.txt)"
        )
        if not fp:
            return
        try:
            body = Path(fp).read_text(encoding="utf-8")
            self.predictions = parse_predictions_yolo_txt(
                body,
                object_list=self.object_list,
                image_w=self.origin_width,
                image_h=self.origin_height,
            )
        except (OSError, ValueError, IndexError) as e:
            QtWidgets.QMessageBox.critical(self, "Load predictions", str(e))
            return
        self._refresh_pred_listwidget()
        self.set_img_ratio()

    def clear_predictions(self) -> None:
        self.predictions.clear()
        self._refresh_pred_listwidget()
        self.set_img_ratio()

    def accept_selected_prediction(self) -> None:
        row = self.pred_listwidget.currentRow()
        if row < 0 or row >= len(self.predictions):
            return
        pred = self.predictions[row]
        if pred.pred_status not in (STATUS_PREDICTED, STATUS_EDITED):
            return
        payload = self._gt_actions.build_add_box_from_prediction(pred)
        if payload is None:
            return
        self.predictions.pop(row)
        self._refresh_pred_listwidget()
        n = self.pred_listwidget.count()
        if n:
            self.pred_listwidget.setCurrentRow(min(row, n - 1))
        else:
            self.pred_listwidget.setCurrentRow(-1)
        self._gt_actions.add_box(*payload)
        # Redraw via AddBoxCommand._refresh_canvas (no duplicate orange pred).

    def reject_selected_prediction(self) -> None:
        row = self.pred_listwidget.currentRow()
        if row < 0 or row >= len(self.predictions):
            return
        self.predictions.pop(row)
        self._refresh_pred_listwidget()
        n = self.pred_listwidget.count()
        if n:
            self.pred_listwidget.setCurrentRow(min(row, n - 1))
        else:
            self.pred_listwidget.setCurrentRow(-1)
        self.set_img_ratio()

    # --- Tile view -----------------------------------------------------------

    def _recompute_tile_grid(self) -> None:
        ow = int(getattr(self, "origin_width", 0) or 0)
        oh = int(getattr(self, "origin_height", 0) or 0)
        if ow <= 0 or oh <= 0:
            self._tile_grid = []
            self._tile_panel.set_tile_count(0)
            return
        cfg = TileConfig(
            tile_size=self._tile_panel.tile_size(),
            tile_stride=self._tile_panel.tile_stride(),
        )
        self._tile_grid = compute_tile_grid(ow, oh, cfg)
        self._tile_panel.set_tile_count(len(self._tile_grid))

    def _on_tile_config_changed(self) -> None:
        self._recompute_tile_grid()
        if self._tile_panel.is_enabled():
            self.set_img_ratio()

    def _on_tile_index_changed(self, _idx: int) -> None:
        if self._tile_panel.is_enabled():
            self.set_img_ratio()

    def _on_tile_view_toggled(self, enabled: bool) -> None:
        if enabled:
            self._recompute_tile_grid()
        self.set_img_ratio()

    def run_error_analysis(self) -> None:
        if not self.real_data and not self.predictions:
            QtWidgets.QMessageBox.information(
                self, "Error analysis", "No GT annotations or predictions loaded."
            )
            return
        dlg = ErrorAnalysisDialog(
            self,
            gt_boxes=self.gt_document.gt_boxes(),
            predictions=self.predictions,
            image_id=self.imgfilePath or "",
        )
        dlg.exec()

    def show_statistics(self) -> None:
        if not self.real_data:
            QtWidgets.QMessageBox.information(
                self, "Statistics", "No annotations to analyze."
            )
            return
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        ow = getattr(self, "origin_width", None)
        oh = getattr(self, "origin_height", None)
        recs = self.gt_document.build_metadata_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            object_list=self.object_list,
            class_id_to_super=sup,
        )
        dlg = StatisticsDialog(self, records=recs)
        dlg.exec()

    def chooseImg(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='rembg_img', filter='IMAGE(*.jpg *.png *.gif *.bmp)'
        )
        if filePath:
            self.resetVal()
            self.Hflip.setDisabled(False)
            self.Hflip.setChecked(False)
            self._paste_candidate_controller.load_asset(filePath)

    def Hflippimg(self, cb):
        self._paste_candidate_controller.set_horizontal_flip(cb.isChecked())

    def inputPimg(self):
        candidate = self.paste_candidate
        self._paste_actions.prompt_add_candidate(
            bbox_row=candidate.bbox_pimg,
            real_bbox_row=candidate.real_bbox_pimg,
            paste_image=candidate.norm_pimg,
            preview_canvas=candidate.pasteimg_canvas,
        )
        candidate.clear_candidate()

    def _refresh_paste_canvas_from_committed(self) -> None:
        try:
            pm = self.origin_canvas.scaledToHeight(self.qpixmap_height)
            draw_paste_images_on_canvas(pm, self.paste_images)
            self._image_canvas.set_canvas(pm)
            self.update()
            self.hideBox.setChecked(False)
            self.hideBbox(self.hideBox)
        except (AttributeError, ZeroDivisionError):
            return

    def _build_paste_record(
        self,
        class_name: str,
        real_row: list[object],
    ) -> PasteRecord | None:
        try:
            bbox_x1, bbox_y1, bbox_x2, bbox_y2 = bbox_from_legacy_paste_row(real_row)
            scale = pow(10, (self.slider_1.value() - 50) / 50)
            return PasteRecord(
                image_path=self.imgfilePath or "",
                asset_path=self.paste_candidate.asset_path,
                class_name=class_name,
                scale=round(scale, 4),
                rotation_deg=float(self.slider_2.value()),
                h_flip=self.Hflip.isChecked(),
                brightness=self.slider_3.value(),
                contrast=self.slider_4.value(),
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
        self._set_paste_adjustment_labels('100 %', '0 °', '100', '100')

    def _get_paste_adjustments(self) -> tuple[int, int, int, int]:
        return (
            self.slider_1.value(),
            self.slider_2.value(),
            self.slider_3.value(),
            self.slider_4.value(),
        )

    def _set_paste_adjustment_labels(
        self,
        scale_text: str,
        rotate_text: str,
        brightness_text: str,
        contrast_text: str,
    ) -> None:
        self.label_val_1.setText(scale_text)
        self.label_val_2.setText(rotate_text)
        self.label_val_3.setText(brightness_text)
        self.label_val_4.setText(contrast_text)

    def _prepare_paste_mode(self) -> None:
        self._gt_edit.cancel_active_drag()
        self._gt_draw.clear_pending()
        self.hideBox.setChecked(True)
        self.hideBbox(self.hideBox)

    def controlpimg(self):
        self._paste_candidate_controller.recompute_preview()

    def pasteImg(self):
        self._paste_candidate_controller.enter_paste_mode()

    def newFile(self):
        filepath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='dataset', filter='IMAGE(*.jpg *.png *.gif *.bmp)'
        )
        if filepath:
            ret = self.mbox.question(
                self, 'question', 'New File?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                self.imgfilePath = filepath
                self.setWindowTitle('ImgLab and ImgBlending   ' + filepath)
                self.img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)

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

                if self.img.ndim == 2:
                    img = cv2.cvtColor(self.img, cv2.COLOR_GRAY2RGB)
                elif self.img.ndim == 3:
                    img = cv2.cvtColor(self.img, cv2.COLOR_BGR2RGB)

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
                self._project_config.add_recent_image(filepath)
                self._restart_autosave_timer()
                self._check_autosave_recovery(filepath)

    def inputObj(self):
        ClassMappingDialog(self, self).exec()

    def loadLabel(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='dataset', filter='TXT (*.txt)'
        )
        if filePath:
            ret = self.mbox.question(
                self, 'question', 'New BoundingBox Label File?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                try:
                    mapping = class_mapping_from_object_list(self.object_list)
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
                    if blocks:
                        self._gt_actions.append_blocks(blocks)
                    else:
                        self.gt_list_view.set_total(len(self.real_data))
                except (OSError, ValueError, IndexError):
                    self.mbox = QtWidgets.QMessageBox(self)
                    self.mbox.setText('This file is not in YOLO label format!')
                    self.mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                    self.mbox.exec()

    def showLabel(self):
        showimg = self.origin_canvas.copy()
        draw_paste_images_on_canvas(showimg, self.paste_images)
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
