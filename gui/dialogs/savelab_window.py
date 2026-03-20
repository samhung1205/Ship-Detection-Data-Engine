"""
Save label dialog with format selection.
"""
from PyQt6 import QtWidgets

from sdde.autosave import remove_autosave


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
        self.box_format.addItems(['YOLO(v5~10)', 'Bounding Boxes'])
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
        default_name = m.imgfilePath.split('/')[-1][:-4] if m.imgfilePath else 'labels'
        self.format = self.box_format.currentText()
        if self.format == 'YOLO(v5~10)':
            filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                self, directory=default_name + '.txt', filter='TXT(*.txt)'
            )
            if filePath:
                with open(filePath, 'w') as file:
                    for data in m.real_data + m.real_pimg_data:
                        name, x1, y1, x2, y2 = data
                        numobj = m.object_list.index(name)
                        n_x = (x1 + x2) / 2.0 / m.origin_width
                        n_y = (y1 + y2) / 2.0 / m.origin_height
                        n_w = (x2 - x1) / m.origin_width
                        n_h = (y2 - y1) / m.origin_height
                        file.write('%s %s %s %s %s\n' % (numobj, n_x, n_y, n_w, n_h))
                self._on_save_success()
                self.close()
        else:
            filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                self, directory=default_name + '_bbox.txt', filter='TXT(*.txt)'
            )
            if filePath:
                with open(filePath, 'w') as file:
                    for data in m.real_data + m.real_pimg_data:
                        name, x1, y1, x2, y2 = data
                        numobj = m.object_list.index(name)
                        file.write('%s %s %s %s %s\n' % (x1, y1, x2, y2, numobj))
                self._on_save_success()
                self.close()

    def _on_save_success(self) -> None:
        if self.main_widget.imgfilePath:
            remove_autosave(self.main_widget.imgfilePath)
        if hasattr(self.main_widget, "_update_autosave_status"):
            self.main_widget._update_autosave_status("Saved (autosave cleared)")

    def closeWindow(self):
        self.close()
