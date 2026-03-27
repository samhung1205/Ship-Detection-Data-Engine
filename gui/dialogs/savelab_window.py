"""
Save label dialog with format selection.
"""
from pathlib import Path

from PyQt6 import QtWidgets

from sdde.autosave import remove_autosave
from sdde.import_export import export_bbox_txt, export_yolo_hbb_txt
from sdde.legacy_rows import annotations_from_legacy_rows, class_mapping_from_object_list


class SavelabWindow(QtWidgets.QWidget):
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

        self.format = 'JPG'

        self.box_format = QtWidgets.QComboBox(self)
        self.box_format.addItems(['YOLO', 'Bounding Boxes'])
        self.box_format.setGeometry(10, 30, 150, 30)

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
        try:
            mapping = class_mapping_from_object_list(m.object_list)
            annotations = annotations_from_legacy_rows(
                m.real_data + m.real_pimg_data,
                object_list=m.object_list,
            )

            if self.format == 'YOLO':
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self, directory=default_name + '.txt', filter='TXT(*.txt)'
                )
                if not filePath:
                    return
                body = export_yolo_hbb_txt(
                    annotations,
                    class_mapping=mapping,
                    image_w=m.origin_width,
                    image_h=m.origin_height,
                )
            else:
                filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                    self, directory=default_name + '_bbox.txt', filter='TXT(*.txt)'
                )
                if not filePath:
                    return
                body = export_bbox_txt(
                    annotations,
                    class_mapping=mapping,
                    cls_mode='class_id',
                )

            Path(filePath).write_text(body, encoding='utf-8')
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
