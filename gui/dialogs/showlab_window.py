"""
Show object bounding boxes dialog.
"""
import numpy as np
from PyQt6 import QtWidgets
from PyQt6.QtGui import QPainter, QPen, QColor
from PyQt6.QtGui import QPixmap


class ShowlabWindow(QtWidgets.QWidget):
    def __init__(self, main_widget: 'MyWidget', showimg: QPixmap):
        super().__init__()
        self.setWindowTitle('Show Object Bounding Box')
        self.resize(1100, 700)
        self.main_widget = main_widget
        self.showimg = showimg
        self.ui()

    def ui(self):
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(10, 10, 852, 602)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.scrollArea = QtWidgets.QScrollArea(self.verticalLayoutWidget)
        self.scrollArea.setWidgetResizable(True)

        self.pixmap = QtWidgets.QLabel()
        self.pixmap.setGeometry(0, 0, 852, 602)
        self.scrollArea.setWidget(self.pixmap)
        self.verticalLayout.addWidget(self.scrollArea)

        self.label = QtWidgets.QLabel(self)
        self.label.setText('Choose object:')
        self.label.setGeometry(888, 95, 100, 10)

        self.box = QtWidgets.QComboBox(self)
        self.box.addItems(self.main_widget.object_list)
        self.box.insertItem(0, 'All')
        self.box.setCurrentIndex(0)
        self.box.setGeometry(880, 110, 200, 30)
        self.box.currentIndexChanged.connect(self.showobjlab)

        np.random.seed(12)
        obj_list = self.main_widget.object_list
        self.colors = [[np.random.randint(0, 255) for _ in range(3)] for _ in obj_list]

        copy_showimg = self.showimg.copy()
        qpainter = QPainter()
        qpainter.begin(copy_showimg)
        for datas in self.main_widget.real_data + self.main_widget.real_pimg_data:
            name, x1, y1, x2, y2 = datas
            r, g, b = self.colors[obj_list.index(name)]
            qpainter.setPen(QPen(QColor(r, g, b), 1))
            qpainter.drawRect(int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1)))
        qpainter.end()
        self.pixmap.setPixmap(copy_showimg)

    def showobjlab(self):
        copy_showimg = self.showimg.copy()
        text = self.box.currentText()
        num = self.box.currentIndex()
        obj_list = self.main_widget.object_list
        all_data = self.main_widget.real_data + self.main_widget.real_pimg_data

        if num == 0:
            qpainter = QPainter()
            qpainter.begin(copy_showimg)
            for datas in all_data:
                name, x1, y1, x2, y2 = datas
                r, g, b = self.colors[obj_list.index(name)]
                qpainter.setPen(QPen(QColor(r, g, b), 1))
                qpainter.drawRect(int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1)))
            qpainter.end()
            self.pixmap.setPixmap(copy_showimg)
            self.update()
        else:
            qpainter = QPainter()
            qpainter.begin(copy_showimg)
            for datas in all_data:
                name, x1, y1, x2, y2 = datas
                if name == text:
                    r, g, b = self.colors[obj_list.index(name)]
                    qpainter.setPen(QPen(QColor(r, g, b), 1))
                    qpainter.drawRect(int(x1), int(y1), abs(int(x2 - x1)), abs(int(y2 - y1)))
            qpainter.end()
            self.pixmap.setPixmap(copy_showimg)
            self.update()
