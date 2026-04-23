"""
Save image dialog with format selection.
"""
from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class SaveimgWindow(QtWidgets.QWidget):
    def __init__(self, main_widget: 'MyWidget'):
        super().__init__()
        self.setWindowTitle('Select Savable Format')
        self.resize(240, 180)
        self.main_widget = main_widget
        self.ui()

    def ui(self):
        self.label_format = QtWidgets.QLabel(self)
        self.label_format.setGeometry(10, 0, 100, 30)
        self.label_format.setText('Format')

        self.format = 'JPG'

        self.box_format = QtWidgets.QComboBox(self)
        self.box_format.addItems(['JPG', 'PNG', 'BMP'])
        self.box_format.setGeometry(10, 30, 100, 30)
        self.box_format.currentIndexChanged.connect(self.changeFormat)

        self.label_jpg = QtWidgets.QLabel(self)
        self.label_jpg.setGeometry(10, 60, 150, 30)
        self.label_jpg.setText('JPG Compression quality')

        self.val = 90

        self.label_jpg_val = QtWidgets.QLabel(self)
        self.label_jpg_val.setGeometry(120, 90, 100, 30)
        self.label_jpg_val.setText(str(self.val))

        self.slider = QtWidgets.QSlider(self)
        self.slider.setOrientation(Qt.Orientation.Horizontal)
        self.slider.setGeometry(10, 90, 100, 30)
        self.slider.setRange(0, 100)
        self.slider.setValue(self.val)
        self.slider.valueChanged.connect(self.changeVal)

        self.btn_ok = QtWidgets.QPushButton(self)
        self.btn_ok.setText('OK')
        self.btn_ok.setGeometry(125, 130, 90, 30)
        self.btn_ok.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #FFF;
                background: #0066FF;
                border-radius: 5px;
            }
        ''')
        self.btn_ok.clicked.connect(self.saveImage)

        self.btn_cancel = QtWidgets.QPushButton(self)
        self.btn_cancel.setText('Cancel')
        self.btn_cancel.setGeometry(25, 130, 90, 30)
        self.btn_cancel.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #000;
                background: #E0E0E0;
                border-radius: 5px;
            }
        ''')
        self.btn_cancel.clicked.connect(self.closeWindow)

    def changeFormat(self):
        self.format = self.box_format.currentText()
        if self.format == 'JPG':
            self.label_jpg.setDisabled(False)
            self.label_jpg_val.setDisabled(False)
            self.slider.setDisabled(False)
        else:
            self.label_jpg.setDisabled(True)
            self.label_jpg_val.setDisabled(True)
            self.slider.setDisabled(True)

    def changeVal(self):
        self.val = self.slider.value()
        self.label_jpg_val.setText(str(self.slider.value()))

    def saveImage(self):
        m = self.main_widget
        default_name = m.imgfilePath.split('/')[-1][:-4] if m.imgfilePath else 'image'
        export_pixmap = m.build_export_image_pixmap()
        if export_pixmap is None:
            return
        if self.format == 'JPG':
            filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                self, directory=default_name + '.jpg', filter='JPG(*.jpg)'
            )
            if filePath:
                export_pixmap.save(filePath, quality=self.val)
                self.close()
        elif self.format == 'PNG':
            filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                self, directory=default_name + '.png', filter='PNG(*.png)'
            )
            if filePath:
                export_pixmap.save(filePath, 'png')
                self.close()
        elif self.format == 'BMP':
            filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
                self, directory=default_name + '.bmp', filter='BMP(*.bmp)'
            )
            if filePath:
                export_pixmap.save(filePath, 'bmp')
                self.close()

    def closeWindow(self):
        self.close()
