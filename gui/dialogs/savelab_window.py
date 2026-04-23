"""
Save label dialog with format selection.
"""
from pathlib import Path

from PyQt6 import QtWidgets

from sdde.autosave import remove_autosave
from sdde.import_export import (
    export_bbox_txt,
    export_coco_bbox_json,
    export_pascal_voc_xml,
    export_yolo_hbb_txt,
)
from sdde.legacy_rows import annotations_from_legacy_rows, class_mapping_from_object_list


class SavelabWindow(QtWidgets.QWidget):
    _FORMAT_LABELS = {
        "yolo_hbb": "YOLO",
        "bbox_txt": "Bounding Boxes",
        "coco_json": "COCO JSON",
        "voc_xml": "Pascal VOC XML",
    }
    _LABEL_TO_KEY = {label: key for key, label in _FORMAT_LABELS.items()}

    def __init__(self, main_widget: 'MyWidget'):
        super().__init__()
        self.setWindowTitle('Select Savable Format')
        self.resize(240, 150)
        self.main_widget = main_widget
        self.ui()

    def ui(self):
        self.label_format = QtWidgets.QLabel(self)
        self.label_format.setGeometry(10, 0, 100, 30)
        self.label_format.setText('Format')

        self.format = 'YOLO'

        self.box_format = QtWidgets.QComboBox(self)
        self.box_format.addItems(list(self._FORMAT_LABELS.values()))
        self.box_format.setGeometry(10, 30, 180, 30)
        default_label = self._FORMAT_LABELS.get(
            self.main_widget._savelab_default_format_key(),
            "YOLO",
        )
        self.box_format.setCurrentText(default_label)

        self.btn_ok = QtWidgets.QPushButton(self)
        self.btn_ok.setText('OK')
        self.btn_ok.setGeometry(125, 100, 90, 30)
        self.btn_ok.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #FFF;
                background: #0066FF;
                border-radius: 5px;
            }
        ''')
        self.btn_ok.clicked.connect(self.saveLabel)

        self.btn_cancel = QtWidgets.QPushButton(self)
        self.btn_cancel.setText('Cancel')
        self.btn_cancel.setGeometry(25, 100, 90, 30)
        self.btn_cancel.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #000;
                background: #E0E0E0;
                border-radius: 5px;
            }
        ''')
        self.btn_cancel.clicked.connect(self.closeWindow)

    def saveLabel(self):
        m = self.main_widget
        default_name = Path(m.imgfilePath).stem if m.imgfilePath else 'labels'
        self.format = self.box_format.currentText()
        format_key = self._LABEL_TO_KEY.get(self.format, "yolo_hbb")
        try:
            mapping = class_mapping_from_object_list(m.object_list)
            annotations = annotations_from_legacy_rows(
                m.real_data + m.real_pimg_data,
                object_list=m.object_list,
            )

            if self.format == 'YOLO':
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    directory=str(m._default_label_export_path("yolo_hbb")),
                    filter='TXT(*.txt)',
                )
                if not filePath:
                    return
                body = export_yolo_hbb_txt(
                    annotations,
                    class_mapping=mapping,
                    image_w=m.origin_width,
                    image_h=m.origin_height,
                )
            elif self.format == 'Bounding Boxes':
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    directory=str(m._default_label_export_path("bbox_txt")),
                    filter='TXT(*.txt)',
                )
                if not filePath:
                    return
                body = export_bbox_txt(
                    annotations,
                    class_mapping=mapping,
                    cls_mode='class_id',
                )
            elif self.format == 'COCO JSON':
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    directory=str(m._default_label_export_path("coco_json")),
                    filter='JSON(*.json)',
                )
                if not filePath:
                    return
                body = export_coco_bbox_json(
                    annotations,
                    class_mapping=mapping,
                    image_w=m.origin_width,
                    image_h=m.origin_height,
                    image_path=m.imgfilePath or None,
                )
            else:
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self,
                    directory=str(m._default_label_export_path("voc_xml")),
                    filter='XML(*.xml)',
                )
                if not filePath:
                    return
                body = export_pascal_voc_xml(
                    annotations,
                    class_mapping=mapping,
                    image_w=m.origin_width,
                    image_h=m.origin_height,
                    image_path=m.imgfilePath or None,
                )

            Path(filePath).parent.mkdir(parents=True, exist_ok=True)
            Path(filePath).write_text(body, encoding='utf-8')
            m._project_config.default_export_format = format_key
            self._on_save_success()
            self.close()
        except (OSError, ValueError, IndexError) as e:
            QtWidgets.QMessageBox.critical(self, 'Save failed', str(e))

    def _on_save_success(self) -> None:
        if self.main_widget.imgfilePath:
            remove_autosave(self.main_widget.imgfilePath)
        if hasattr(self.main_widget, "_update_autosave_status"):
            self.main_widget._update_autosave_status("Saved (autosave cleared)")

    def closeWindow(self):
        self.close()
