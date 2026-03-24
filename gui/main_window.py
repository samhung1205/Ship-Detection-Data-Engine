"""
Main application window for ImgLab and ImgBlending.
"""
from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtGui import (
    QAction, QPixmap, QImage, QPainter, QPen, QColor, QCloseEvent,
)
from PyQt6.QtCore import Qt, QRect, QPoint, QTimer
import numpy as np
import cv2

from .constants import (
    STYLE_BUTTON_PRIMARY,
    STYLE_BUTTON_SECONDARY,
    STYLE_BUTTON_SECONDARY_DISABLED,
    STYLE_LIST_WIDGET,
)
from .canvas_utils import draw_bboxes_on_canvas, draw_paste_images_on_canvas
from .canvas_widget import ImageCanvasWidget
from .annotation_controller import (
    AddBoxCommand,
    AnnotationController,
    BulkAppendBoxesCommand,
    ClearAllBoxesCommand,
    RemoveBoxCommand,
    RenameBoxCommand,
)
from sdde.class_catalog import ClassCatalog

from .class_mapping_service import load_class_catalog
from .attribute_panel import AttributePanel
from .dialogs import ClassMappingDialog, ErrorAnalysisDialog, StatisticsDialog, ShowlabWindow, SaveimgWindow, SavelabWindow
from .tile_panel import TilePanel
from sdde.tile import TileConfig, TileRect, compute_tile_grid
from sdde.metadata_export import (
    build_annotation_records,
    export_annotations_csv,
    export_annotations_json,
)
from sdde.prediction import (
    STATUS_EDITED,
    STATUS_PREDICTED,
    parse_predictions_yolo_txt,
)
from sdde.augmentation import (
    PasteRecord,
    export_paste_records_csv,
    export_paste_records_json,
)
from sdde.project_config import ProjectConfig, load_project_config, save_project_config
from sdde.autosave import has_autosave, read_autosave, remove_autosave, write_autosave


class MyWidget(QtWidgets.QWidget):
    def __init__(self, is_confirm_quit: bool = True):
        super().__init__()
        self.setWindowTitle('Ship Detection Data Engine')
        self.resize(1460, 720)
        self.setUpdatesEnabled(True)
        self.is_confirm_quit = is_confirm_quit
        self.x, self.y = None, None
        self.last_x, self.last_y = None, None
        self.ith = None
        self.object_list = []
        self.data = []
        self.real_data = []
        self.pimg_data = []
        self.paste_images = []
        self.real_pimg_data = []
        self.imgfilePath = ''
        self.box_attributes: list[dict] = []
        self.predictions: list = []
        self._tile_grid: list[TileRect] = []
        self.paste_records: list[PasteRecord] = []
        self._current_asset_path: str = ""
        self._project_config = ProjectConfig()
        self._autosave_timer = QTimer(self)
        self._autosave_timer.timeout.connect(self._do_autosave)
        self.ui()
        self.adjustUi()
        self._bootstrap_class_catalog()
        self._annotation_controller = AnnotationController(self)
        self._try_load_default_project_config()

    @property
    def canvas(self):
        """Current display pixmap (scaled + overlays); owned by ImageCanvasWidget."""
        return self._image_canvas.canvas

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
        """Same enable set as legacy InputWindow after saving class names."""
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
            self._annotation_controller.apply(BulkAppendBoxesCommand(blocks))
        if ba:
            for i, attr in enumerate(ba):
                if i < len(self.box_attributes):
                    self.box_attributes[i] = dict(attr)
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

    def append_box_attributes_row(self) -> None:
        """Append default attributes + auto size_tag for last real_data row (undo stack calls this)."""
        from sdde.attributes import compute_size_tag, default_attributes_dict

        i = len(self.real_data) - 1
        if i < 0:
            return
        row = self.real_data[i]
        d = default_attributes_dict()
        x1, y1, x2, y2 = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        d["size_tag"] = compute_size_tag(x1, y1, x2, y2)
        self.box_attributes.append(d)

    def _on_list_box_row_changed(self, row: int) -> None:
        self._refresh_attribute_panel_for_row(row)

    def _refresh_attribute_panel_for_row(self, row: int) -> None:
        from sdde.attributes import default_attributes_dict

        if row < 0 or row >= len(self.box_attributes):
            self._attr_panel.set_enabled_editing(False)
            self._attr_panel.load_from_dict(default_attributes_dict())
            return
        self._attr_panel.set_enabled_editing(True)
        self._attr_panel.load_from_dict(self.box_attributes[row])

    def _on_attr_panel_changed(self) -> None:
        r = self.listwidget.currentRow()
        if r < 0 or r >= len(self.box_attributes):
            return
        self.box_attributes[r] = self._attr_panel.to_dict()

    def _on_recalc_size_tag_for_selection(self) -> None:
        r = self.listwidget.currentRow()
        if r < 0 or r >= len(self.real_data) or r >= len(self.box_attributes):
            return
        from sdde.attributes import compute_size_tag

        row = self.real_data[r]
        x1, y1, x2, y2 = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
        st = compute_size_tag(x1, y1, x2, y2)
        self._attr_panel.combo_size.setCurrentText(st)
        self.box_attributes[r] = self._attr_panel.to_dict()

    def export_annotation_metadata_json(self) -> None:
        sup = {c.class_id: c.super_category for c in self.class_catalog.classes}
        ow = getattr(self, "origin_width", None)
        oh = getattr(self, "origin_height", None)
        recs = build_annotation_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            real_data=self.real_data,
            box_attributes=self.box_attributes,
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
        recs = build_annotation_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            real_data=self.real_data,
            box_attributes=self.box_attributes,
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
        self._image_canvas.set_mouse_move_handler(self.get_position)
        self._image_canvas.set_mouse_press_handler(self.get_clicked_position)

        ### control zoom ###
        self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_in.setGeometry(QRect(155, 520, 89, 25))
        self.btn_zoom_in.setText("zoom_in")
        self.btn_zoom_in.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_in.setDisabled(True)
        self.btn_zoom_in.clicked.connect(self.set_zoom_in)

        self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.slider_zoom.setGeometry(QRect(255, 520, 231, 21))
        self.slider_zoom.setProperty("value", 50)
        self.slider_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.slider_zoom.setDisabled(True)
        self.slider_zoom.valueChanged.connect(self.getslidervalue)

        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QRect(615, 520, 300, 21))

        self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_out.setGeometry(QRect(505, 520, 89, 25))
        self.btn_zoom_out.setText("zoom_out")
        self.btn_zoom_out.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_out.setDisabled(True)
        self.btn_zoom_out.clicked.connect(self.set_zoom_out)

        ### mouseMove（由 ImageCanvasWidget 轉發）###
        self.label_get_pos = QtWidgets.QLabel(self)
        self.label_get_pos.setGeometry(125, 550, 250, 18)
        self.label_get_pos.setText('current position = (x,y)')
        self.label_get_pos.setStyleSheet('font-size: 12px;')

        ### mousePress（由 ImageCanvasWidget 轉發；標註模式會改為 paint / paste）###
        self.label_click_pos = QtWidgets.QLabel(self)
        self.label_click_pos.setGeometry(380, 550, 250, 18)
        self.label_click_pos.setText('clicked position = (x,y)')
        self.label_click_pos.setStyleSheet('font-size: 12px;')

        ### show img.shape ###
        self.label_img_shape = QtWidgets.QLabel(self)
        self.label_img_shape.setGeometry(630, 550, 500, 18)

        self.lbl_autosave_status = QtWidgets.QLabel(self)
        self.lbl_autosave_status.setGeometry(125, 572, 300, 16)
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
        self.listwidget.clicked.connect(self.showObject)
        self.listwidget.currentRowChanged.connect(self._on_list_box_row_changed)
        self.listwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.listwidget.customContextMenuRequested.connect(self.on_context_menu_labimg)

        self._attr_panel = AttributePanel(self)
        self._attr_panel.setGeometry(1010, 212, 430, 170)
        self._attr_panel.values_changed.connect(self._on_attr_panel_changed)
        self._attr_panel.set_recalc_size_callback(self._on_recalc_size_tag_for_selection)
        self._attr_panel.set_enabled_editing(False)

        self.label_clear = QtWidgets.QPushButton(self)
        self.label_clear.setText('Delete all')
        self.label_clear.setGeometry(1380, 386, 60, 20)
        self.label_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.label_clear.clicked.connect(self.allbboxClear)

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
        self.btn_chooseimg.setGeometry(1010, 550, 50, 20)
        self.btn_chooseimg.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_chooseimg.clicked.connect(self.chooseImg)

        self.btn_add = QtWidgets.QPushButton(self)
        self.btn_add.setText('Add')
        self.btn_add.setGeometry(1065, 550, 50, 20)
        self.btn_add.setStyleSheet(STYLE_BUTTON_SECONDARY_DISABLED)
        self.btn_add.setDisabled(True)
        self.btn_add.clicked.connect(self.inputPimg)

        self.btn_reset = QtWidgets.QPushButton(self)
        self.btn_reset.setText('Reset')
        self.btn_reset.setGeometry(1390, 550, 50, 20)
        self.btn_reset.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_reset.clicked.connect(self.resetVal)

        ### Paste_imgae_QListWidget ###
        self.pimg_list = QtWidgets.QLabel(self)
        self.pimg_list.setText('Paste Images')
        self.pimg_list.setGeometry(1010, 574, 200, 20)
        self.pimg_list.setStyleSheet('font-size: 12px;')

        self.pimglistwidget = QtWidgets.QListWidget(self)
        self.pimglistwidget.addItems([])
        self.pimglistwidget.setGeometry(1010, 596, 430, 74)
        self.pimglistwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.pimglistwidget.clicked.connect(self.showPimg)
        self.pimglistwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pimglistwidget.customContextMenuRequested.connect(self.on_context_menu_pasteimg)

        self.pimg_clear = QtWidgets.QPushButton(self)
        self.pimg_clear.setText('Delete all')
        self.pimg_clear.setGeometry(1380, 674, 60, 20)
        self.pimg_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.pimg_clear.clicked.connect(self.allpimgClear)

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
        self.slider_1.setGeometry(sx + 80, 456, sw, 16)
        self.slider_1.setRange(0, 100)
        self.slider_1.setValue(50)
        self.slider_1.valueChanged.connect(self.controlpimg)
        self.label_val_1 = QtWidgets.QLabel(self)
        self.label_val_1.setGeometry(sx + 265, 456, 50, 16)
        self.label_val_1.setText("100 %")
        self.label_val_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_2 = QtWidgets.QLabel(self)
        self.label_adj_2.setGeometry(sx, 480, 70, 14)
        self.label_adj_2.setText('Rotate')
        self.label_adj_2.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_2 = QtWidgets.QSlider(self)
        self.slider_2.setOrientation(Qt.Orientation.Horizontal)
        self.slider_2.setGeometry(sx + 80, 480, sw, 16)
        self.slider_2.setRange(0, 360)
        self.slider_2.setValue(0)
        self.slider_2.valueChanged.connect(self.controlpimg)
        self.label_val_2 = QtWidgets.QLabel(self)
        self.label_val_2.setGeometry(sx + 265, 480, 50, 16)
        self.label_val_2.setText('0 °')
        self.label_val_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_3 = QtWidgets.QLabel(self)
        self.label_adj_3.setGeometry(sx, 504, 70, 14)
        self.label_adj_3.setText('Brightness')
        self.label_adj_3.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_3 = QtWidgets.QSlider(self)
        self.slider_3.setOrientation(Qt.Orientation.Horizontal)
        self.slider_3.setGeometry(sx + 80, 504, sw, 16)
        self.slider_3.setRange(0, 200)
        self.slider_3.setValue(100)
        self.slider_3.valueChanged.connect(self.controlpimg)
        self.label_val_3 = QtWidgets.QLabel(self)
        self.label_val_3.setGeometry(sx + 265, 504, 50, 16)
        self.label_val_3.setText('100')
        self.label_val_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_adj_4 = QtWidgets.QLabel(self)
        self.label_adj_4.setGeometry(sx, 528, 70, 14)
        self.label_adj_4.setText('Contrast')
        self.label_adj_4.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.slider_4 = QtWidgets.QSlider(self)
        self.slider_4.setOrientation(Qt.Orientation.Horizontal)
        self.slider_4.setGeometry(sx + 80, 528, sw, 16)
        self.slider_4.setRange(0, 200)
        self.slider_4.setValue(100)
        self.slider_4.valueChanged.connect(self.controlpimg)
        self.label_val_4 = QtWidgets.QLabel(self)
        self.label_val_4.setGeometry(sx + 265, 528, 50, 16)
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

    def set_img_ratio(self):
        ow = int(getattr(self, "origin_width", 0) or 0)
        self.ratio_rate, self.qpixmap_height = self._image_canvas.redraw_scaled_overlay(
            origin_canvas=self.origin_canvas,
            ratio_value=self.ratio_value,
            origin_height=self.origin_height,
            origin_width=ow,
            hide_boxes=self.hideBox.isChecked(),
            bbox_data=self.data,
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

    def get_clicked_position(self, event):
        mx = int(event.position().x())
        my = int(event.position().y())
        self.__update_text_clicked_position(mx, my)

    def __update_text_img_shape(self):
        current_text = f"Current img shape = ({self.canvas.width()}, {self.canvas.height()})"
        origin_text = f"Origin img shape = ({self.origin_width}, {self.origin_height})"
        self.label_img_shape.setText(current_text + "\t" + origin_text)

    def make_label(self):
        self.hideBox.setChecked(False)
        self.hideBbox(self.hideBox)
        self._image_canvas.set_mouse_press_handler(self.paint)

    def paint(self, event):
        mx = int(event.position().x())
        my = int(event.position().y())
        self.__update_text_clicked_position(mx, my)

        if mx < self.canvas.width() and my < self.canvas.height():
            qpainter = QPainter()
            qpainter.begin(self.canvas)
            qpainter.setPen(QPen(QColor('#00ff00'), 3))
            qpainter.drawPoint(mx, my)
            qpainter.end()
            self._image_canvas.sync_label_from_canvas()
            self.update()
            if self.x is None and self.y is None:
                self.x, self.y = mx, my
            else:
                if mx > self.x and my > self.y:
                    self.last_x, self.last_y = mx, my
                    qpainter = QPainter()
                    qpainter.begin(self.canvas)
                    qpainter.setPen(QPen(QColor('#00ff00'), 1))
                    qpainter.drawRect(
                        self.x, self.y,
                        abs(self.x - self.last_x), abs(self.y - self.last_y)
                    )
                    qpainter.end()
                    self._image_canvas.sync_label_from_canvas()
                    self.update()
                    self.qInput()
                    self.x, self.y = None, None
                else:
                    self.x, self.y = None, None
                    if self.hideBox.isChecked():
                        self._image_canvas.set_canvas(
                            self.origin_canvas.scaledToHeight(self.qpixmap_height)
                        )
                        self.update()
                    else:
                        pm = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                        draw_bboxes_on_canvas(pm, self.data + self.pimg_data)
                        self._image_canvas.set_canvas(pm)
                        self.update()

    def qInput(self):
        item, ok = QtWidgets.QInputDialog().getItem(
            self, '', 'Enter object name', self.object_list, 0
        )
        if ok:
            real_x = int(self.x * self.origin_width / self.canvas.width())
            real_y = int(self.y * self.origin_height / self.canvas.height())
            real_last_x = int(self.last_x * self.origin_width / self.canvas.width())
            real_last_y = int(self.last_y * self.origin_height / self.canvas.height())
            data_row = [
                item, self.x, self.y, self.last_x, self.last_y,
                self.canvas.width(), self.canvas.height(),
            ]
            real_row = [item, real_x, real_y, real_last_x, real_last_y]
            extended = item not in self.object_list
            self._annotation_controller.apply(
                AddBoxCommand(
                    data_row,
                    real_row,
                    item,
                    extended_object_list=extended,
                )
            )
            self.listwidget.setCurrentRow(self.listwidget.count() - 1)
        else:
            if self.hideBox.isChecked():
                self._image_canvas.set_canvas(
                    self.origin_canvas.scaledToHeight(self.qpixmap_height)
                )
                self.update()
            else:
                pm = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                draw_bboxes_on_canvas(pm, self.data + self.pimg_data)
                self._image_canvas.set_canvas(pm)
                self.update()

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
        canvas = self.canvas
        if canvas is None:
            return
        cw, ch = canvas.width(), canvas.height()
        item = pred.class_name
        mx1 = pred.x1 * cw / self.origin_width
        my1 = pred.y1 * ch / self.origin_height
        mx2 = pred.x2 * cw / self.origin_width
        my2 = pred.y2 * ch / self.origin_height
        data_row = [item, mx1, my1, mx2, my2, cw, ch]
        real_row = [item, int(pred.x1), int(pred.y1), int(pred.x2), int(pred.y2)]
        ext = item not in self.object_list
        self.predictions.pop(row)
        self._refresh_pred_listwidget()
        n = self.pred_listwidget.count()
        if n:
            self.pred_listwidget.setCurrentRow(min(row, n - 1))
        else:
            self.pred_listwidget.setCurrentRow(-1)
        self._annotation_controller.apply(
            AddBoxCommand(data_row, real_row, item, extended_object_list=ext)
        )
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
        gt_boxes: list[tuple[str, float, float, float, float]] = []
        for row in self.real_data:
            name = row[0]
            x1, y1, x2, y2 = float(row[1]), float(row[2]), float(row[3]), float(row[4])
            gt_boxes.append((name, x1, y1, x2, y2))
        dlg = ErrorAnalysisDialog(
            self,
            gt_boxes=gt_boxes,
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
        recs = build_annotation_records(
            image_path=self.imgfilePath or None,
            image_width=ow,
            image_height=oh,
            real_data=self.real_data,
            box_attributes=self.box_attributes,
            object_list=self.object_list,
            class_id_to_super=sup,
        )
        dlg = StatisticsDialog(self, records=recs)
        dlg.exec()

    def showObject(self):
        num = self.listwidget.currentIndex().row()
        self.ith1 = num
        x1, y1, x2, y2, w, h = self.data[num][1:]
        x1 *= self.canvas.width() / w
        y1 *= self.canvas.height() / h
        x2 *= self.canvas.width() / w
        y2 *= self.canvas.height() / h
        copy_canvas = self.canvas.copy()
        qpainter = QPainter()
        qpainter.begin(copy_canvas)
        color = QColor(30, 144, 255, 120)
        qpainter.fillRect(
            int(x1), int(y1),
            abs(int(x2 - x1)) + 1, abs(int(y2 - y1)) + 1,
            color
        )
        qpainter.end()
        self._image_canvas.paint_label_only(copy_canvas)
        self.update()

    def bboxClear(self):
        try:
            self._annotation_controller.apply(RemoveBoxCommand(self.ith1))
        except (IndexError, AttributeError):
            return

    def allbboxClear(self):
        try:
            ret = self.mbox.question(
                self, 'question', 'Delete all?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                self._annotation_controller.apply(ClearAllBoxesCommand())
        except (AttributeError, ZeroDivisionError):
            return

    def bboxRename(self):
        try:
            text, ok = QtWidgets.QInputDialog().getItem(
                self, '', 'Enter object name', self.object_list, 0
            )
            if ok:
                old = self.data[self.ith1][0]
                if old == text:
                    return
                self._annotation_controller.apply(
                    RenameBoxCommand(self.ith1, old, text)
                )
        except (IndexError, AttributeError):
            return

    def chooseImg(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='rembg_img', filter='IMAGE(*.jpg *.png *.gif *.bmp)'
        )
        if filePath:
            self._current_asset_path = filePath
            self.pasteimg = cv2.imread(filePath, cv2.IMREAD_UNCHANGED)
            oW = np.sum(self.pasteimg[:, :, 3], axis=0)
            oW[oW != 0] = 1
            min_x = np.min(np.where(oW == 1))
            max_x = np.max(np.where(oW == 1))
            oH = np.sum(self.pasteimg[:, :, 3], axis=1)
            oH[oH != 0] = 1
            min_y = np.min(np.where(oH == 1))
            max_y = np.max(np.where(oH == 1))
            self.pasteimg = self.pasteimg[min_y:max_y + 1, min_x:max_x + 1, :]
            self.pasteimg = np.pad(
                self.pasteimg, ((1, 1), (1, 1), (0, 0)),
                "constant", constant_values=0
            )
            self.origin_pasteimg = self.pasteimg.copy()
            self.resetVal()
            self.Hflip.setDisabled(False)
            self.Hflip.setChecked(False)
            self.Hflippimg(self.Hflip)

    def Hflippimg(self, cb):
        if cb.isChecked():
            self.pasteimg = self.origin_pasteimg[:, ::-1, :]
        else:
            self.pasteimg = self.origin_pasteimg
        pasteimg = cv2.cvtColor(self.pasteimg, cv2.COLOR_BGRA2RGBA)
        self.pasteimg_height, self.pasteimg_width, self.pasteimg_channel = self.pasteimg.shape
        bytesPerline = self.pasteimg_channel * self.pasteimg_width
        pimg = QImage(
            pasteimg, self.pasteimg_width, self.pasteimg_height,
            bytesPerline, QImage.Format.Format_RGBA8888
        )
        self.paste_canvas = QPixmap.fromImage(pimg)
        if self.pasteimg_width < self.pasteimg_height:
            self.paste_canvas = self.paste_canvas.scaled(
                int(80 * self.pasteimg_width / self.pasteimg_height), 80
            )
        else:
            self.paste_canvas = self.paste_canvas.scaled(
                80, int(80 * self.pasteimg_height / self.pasteimg_width)
            )
        self.pmap_pasteimg.setPixmap(self.paste_canvas)

    def inputPimg(self):
        item, ok = QtWidgets.QInputDialog().getItem(
            self, '', 'Enter object name', self.object_list, 0
        )
        if ok:
            self.pimglistwidget.addItem(item)
            self.bbox_pimg.insert(0, item)
            self.real_bbox_pimg.insert(0, item)
            if item not in self.object_list:
                self.object_list.append(item)
            self.pimg_data.append(self.bbox_pimg)
            self.real_pimg_data.append(self.real_bbox_pimg)
            self.paste_images.append(self.norm_pimg)
            self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
            self._image_canvas.set_canvas(self.pasteimg_canvas)
            self.btn_add.setDisabled(True)
            self._record_paste(item)
        else:
            self._image_canvas.sync_label_from_canvas()
        self.cX, self.cY = None, None

    def showPimg(self):
        num = self.pimglistwidget.currentIndex().row()
        self.ith2 = num
        x1, y1, x2, y2, w, h = self.pimg_data[num][1:]
        x1 *= self.canvas.width() / w
        y1 *= self.canvas.height() / h
        x2 *= self.canvas.width() / w
        y2 *= self.canvas.height() / h
        copy_canvas = self.canvas.copy()
        qpainter = QPainter()
        qpainter.begin(copy_canvas)
        color = QColor(30, 144, 255, 120)
        qpainter.fillRect(
            int(x1), int(y1),
            abs(int(x2 - x1)) + 1, abs(int(y2 - y1)) + 1,
            color
        )
        qpainter.end()
        self._image_canvas.paint_label_only(copy_canvas)
        self.update()

    def pimgClear(self):
        try:
            self.pimg_data.pop(self.ith2)
            self.real_pimg_data.pop(self.ith2)
            self.paste_images.pop(self.ith2)
            if self.ith2 < len(self.paste_records):
                self.paste_records.pop(self.ith2)
            self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
            self.pimglistwidget.takeItem(self.ith2)
            pm = self.origin_canvas.scaledToHeight(self.qpixmap_height)
            draw_paste_images_on_canvas(pm, self.paste_images)
            self._image_canvas.set_canvas(pm)
            self.update()
            self.hideBox.setChecked(False)
            self.hideBbox(self.hideBox)
        except (IndexError, AttributeError):
            return

    def allpimgClear(self):
        try:
            ret = self.mbox.question(
                self, 'question', 'Delete all?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                self.pimg_data.clear()
                self.real_pimg_data.clear()
                self.paste_images.clear()
                self.paste_records.clear()
                self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
                self.pimglistwidget.clear()
                self._image_canvas.set_canvas(
                    self.origin_canvas.scaledToHeight(self.qpixmap_height)
                )
                self.update()
                self.hideBox.setChecked(False)
                self.hideBbox(self.hideBox)
        except (AttributeError, ZeroDivisionError):
            return

    def pimgRename(self):
        try:
            text, ok = QtWidgets.QInputDialog().getItem(
                self, '', 'Enter object name', self.object_list, 0
            )
            if ok:
                self.pimg_data[self.ith2][0] = text
                self.real_pimg_data[self.ith2][0] = text
                item = self.pimglistwidget.item(self.ith2)
                item.setText(text)
                if text not in self.object_list:
                    self.object_list.append(text)
        except (IndexError, AttributeError):
            return

    def _record_paste(self, class_name: str) -> None:
        """Capture current paste transform params into a PasteRecord."""
        try:
            scale = pow(10, (self.slider_1.value() - 50) / 50)
            rec = PasteRecord(
                image_path=self.imgfilePath or "",
                asset_path=self._current_asset_path,
                class_name=class_name,
                scale=round(scale, 4),
                rotation_deg=float(self.slider_2.value()),
                h_flip=self.Hflip.isChecked(),
                brightness=self.slider_3.value(),
                contrast=self.slider_4.value(),
                bbox_x1=int(self.real_bbox_pimg[0]),
                bbox_y1=int(self.real_bbox_pimg[1]),
                bbox_x2=int(self.real_bbox_pimg[2]),
                bbox_y2=int(self.real_bbox_pimg[3]),
            )
            self.paste_records.append(rec)
        except (AttributeError, IndexError):
            pass

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
        self.label_val_1.setText('100 %')
        self.label_val_2.setText('0 °')
        self.label_val_3.setText('100')
        self.label_val_4.setText('100')

    def controlpimg(self):
        val1 = self.slider_1.value()
        val2 = self.slider_2.value()
        val3 = self.slider_3.value()
        val4 = self.slider_4.value()
        rate1 = pow(10, (val1 - 50) / 50)

        try:
            width = int(self.pasteimg_width * rate1)
            height = int(self.pasteimg_height * rate1)
            dim = (width, height)
            self.resizeimg = cv2.resize(self.pasteimg, dim, interpolation=cv2.INTER_AREA)
            rH, rW = self.resizeimg.shape[:2]

            if rH % 2 == 0:
                self.resizeimg = np.pad(
                    self.resizeimg, ((1, 0), (0, 0), (0, 0)),
                    "constant", constant_values=0
                )
                rH += 1
            if rW % 2 == 0:
                self.resizeimg = np.pad(
                    self.resizeimg, ((0, 0), (1, 0), (0, 0)),
                    "constant", constant_values=0
                )
                rW += 1

            (cX, cY) = (rW // 2, rH // 2)
            M = cv2.getRotationMatrix2D((cX, cY), -val2, 1)
            cos = np.abs(M[0, 0])
            sin = np.abs(M[0, 1])
            nW = int((rH * sin) + (rW * cos))
            nH = int((rH * cos) + (rW * sin))
            M[0, 2] += (nW - rW) / 2
            M[1, 2] += (nH - rH) / 2
            self.rotated = cv2.warpAffine(
                self.resizeimg, M, (nW, nH), flags=cv2.INTER_AREA
            )

            b = val3 - 100
            c = max(-99, val4 - 100)
            bc_img = self.rotated[:, :, :3] * (c / 100 + 1) - c + b
            bc_img = np.clip(bc_img, 0, 255)
            bc_img = np.uint8(bc_img)
            self.bc_image = np.dstack((bc_img, self.rotated[:, :, 3]))

            left = self.cX - nW // 2
            top = self.cY - nH // 2
            right = self.cX + nW - nW // 2
            down = self.cY + nH - nH // 2

            self.pasteimg_canvas = self.canvas.copy()
            qpainter = QPainter()
            qpainter.begin(self.pasteimg_canvas)
            pimg = cv2.cvtColor(self.bc_image, cv2.COLOR_BGRA2RGBA)
            bytesPerline = 4 * nW
            pimage = QImage(
                pimg, nW, nH, bytesPerline, QImage.Format.Format_RGBA8888
            )
            qpainter.drawImage(QRect(left, top, nW, nH), pimage)
            qpainter.end()
            self._image_canvas.paint_label_only(self.pasteimg_canvas)
            self.update()

            W = np.sum(self.bc_image[:, :, 3], axis=0)
            W[W != 0] = 1
            x1 = left + np.min(np.where(W == 1)) - 1
            x2 = right - (W.shape[0] - np.max(np.where(W == 1))) + 1
            H = np.sum(self.bc_image[:, :, 3], axis=1)
            H[H != 0] = 1
            y1 = top + np.min(np.where(H == 1)) - 1
            y2 = down - (H.shape[0] - np.max(np.where(H == 1))) + 1

            real_x1 = max(0, int(x1 * self.origin_width / self.pasteimg_canvas.width()))
            real_y1 = max(0, int(y1 * self.origin_height / self.pasteimg_canvas.height()))
            real_x2 = min(
                int(x2 * self.origin_width / self.pasteimg_canvas.width()),
                self.origin_width
            )
            real_y2 = min(
                int(y2 * self.origin_height / self.pasteimg_canvas.height()),
                self.origin_height
            )

            self.norm_pimg = [
                pimage,
                left / self.pasteimg_canvas.width(),
                top / self.pasteimg_canvas.height(),
                nW / self.pasteimg_canvas.width(),
                nH / self.pasteimg_canvas.height(),
            ]
            self.bbox_pimg = [
                x1, y1, x2, y2,
                self.pasteimg_canvas.width(), self.pasteimg_canvas.height(),
            ]
            self.real_bbox_pimg = [real_x1, real_y1, real_x2, real_y2]

        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass

        self.label_val_1.setText(f"{int(100 * rate1)} %")
        self.label_val_2.setText(str(val2) + ' °')
        self.label_val_3.setText(str(val3))
        self.label_val_4.setText(str(val4))

    def pasteImg(self):
        self.hideBox.setChecked(True)
        self.hideBbox(self.hideBox)
        self._image_canvas.set_mouse_press_handler(self.paste)

    def paste(self, event):
        x = int(event.position().x())
        y = int(event.position().y())
        self.__update_text_clicked_position(x, y)
        if x < self.canvas.width() and y < self.canvas.height():
            self.hideBox.setChecked(True)
            self.hideBbox(self.hideBox)
            self.btn_add.setDisabled(False)
            self.cX, self.cY = x, y
            self.controlpimg()

    def on_context_menu_labimg(self, pos):
        context = QtWidgets.QMenu(self)
        self.action_labimgrename = QAction("Rename", self)
        self.action_labimgdelete = QAction("Delete", self)
        context.addAction(self.action_labimgrename)
        context.addAction(self.action_labimgdelete)
        self.action_labimgrename.triggered.connect(self.bboxRename)
        self.action_labimgdelete.triggered.connect(self.bboxClear)
        context.exec(self.mapToGlobal(pos + QPoint(1010, 96)))

    def on_context_menu_pasteimg(self, pos):
        context = QtWidgets.QMenu(self)
        self.action_pimgrename = QAction("Rename", self)
        self.action_pimgdelete = QAction("Delete", self)
        context.addAction(self.action_pimgrename)
        context.addAction(self.action_pimgdelete)
        self.action_pimgrename.triggered.connect(self.pimgRename)
        self.action_pimgdelete.triggered.connect(self.pimgClear)
        context.exec(self.mapToGlobal(pos + QPoint(1010, 596)))

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

                self.paste_images = []
                self.real_data = []
                self.data = []
                self.real_pimg_data = []
                self.pimg_data = []
                self.predictions = []
                self.paste_records = []
                self._current_asset_path = ""
                self._annotation_controller.reset()

                self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')
                self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
                self.listwidget.clear()
                self.pimglistwidget.clear()
                self.box_attributes.clear()
                self._refresh_pred_listwidget()
                self.listwidget.setCurrentRow(-1)
                self._refresh_attribute_panel_for_row(-1)
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
                    blocks = []
                    with open(filePath, encoding='utf-8') as file_obj:
                        for raw in file_obj:
                            line = raw.strip()
                            if not line:
                                continue
                            data = line.split()
                            obj = int(data[0])
                            x, y, w, h = [float(num) for num in data[1:]]
                            x1 = int(x * self.origin_width - w * self.origin_width / 2)
                            y1 = int(y * self.origin_height - h * self.origin_height / 2)
                            x2 = int(x * self.origin_width + w * self.origin_width / 2)
                            y2 = int(y * self.origin_height + h * self.origin_height / 2)
                            obj_name = self.object_list[obj]
                            d_row = [
                                obj_name, x1, y1, x2, y2,
                                self.origin_width, self.origin_height,
                            ]
                            r_row = [obj_name, x1, y1, x2, y2]
                            blocks.append((d_row, r_row, obj_name))
                    if blocks:
                        self._annotation_controller.apply(BulkAppendBoxesCommand(blocks))
                    else:
                        self.label_list.setText(
                            f'Box Labels  (Total: {len(self.real_data)})'
                        )
                except (ValueError, IndexError, KeyError):
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
