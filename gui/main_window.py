"""
Main application window for ImgLab and ImgBlending.
"""
from PyQt6 import QtWidgets
from PyQt6.QtGui import (
    QAction, QPixmap, QImage, QPainter, QPen, QColor, QCloseEvent,
)
from PyQt6.QtCore import Qt, QRect, QPoint
import numpy as np
import cv2

from .constants import (
    STYLE_BUTTON_PRIMARY,
    STYLE_BUTTON_SECONDARY,
    STYLE_BUTTON_SECONDARY_DISABLED,
    STYLE_LIST_WIDGET,
)
from .canvas_utils import draw_bboxes_on_canvas, draw_paste_images_on_canvas
from .dialogs import InputWindow, ShowlabWindow, SaveimgWindow, SavelabWindow


class MyWidget(QtWidgets.QWidget):
    def __init__(self, is_confirm_quit: bool = True):
        super().__init__()
        self.setWindowTitle('ImgLab and ImgBlending')
        self.resize(1300, 770)
        self.setUpdatesEnabled(True)
        self.is_confirm_quit = is_confirm_quit
        self.x, self.y = None, None
        self.last_x, self.last_y = None, None
        self.ith = None
        self.object_list = []
        self.real_data = []
        self.real_pimg_data = []
        self.imgfilePath = ''
        self.ui()
        self.adjustUi()

    def ui(self):
        ### 畫布 ###
        self.centralwidget = QtWidgets.QWidget(self)
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(120, 20, 852, 602)
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setSpacing(0)
        self.scrollArea = QtWidgets.QScrollArea(self.verticalLayoutWidget)
        self.scrollArea.setWidgetResizable(True)

        self.pmap = QtWidgets.QLabel()
        self.pmap.setGeometry(0, 0, 852, 602)
        self.pmap.setCursor(Qt.CursorShape.CrossCursor)
        self.scrollArea.setWidget(self.pmap)
        self.verticalLayout.addWidget(self.scrollArea)

        ### control zoom ###
        self.btn_zoom_in = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_in.setGeometry(QRect(150, 650, 89, 25))
        self.btn_zoom_in.setText("zoom_in")
        self.btn_zoom_in.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_in.setDisabled(True)
        self.btn_zoom_in.clicked.connect(self.set_zoom_in)

        self.slider_zoom = QtWidgets.QSlider(self.centralwidget)
        self.slider_zoom.setGeometry(QRect(250, 650, 231, 21))
        self.slider_zoom.setProperty("value", 50)
        self.slider_zoom.setOrientation(Qt.Orientation.Horizontal)
        self.slider_zoom.setDisabled(True)
        self.slider_zoom.valueChanged.connect(self.getslidervalue)

        self.label_ratio = QtWidgets.QLabel(self.centralwidget)
        self.label_ratio.setGeometry(QRect(610, 650, 641, 21))

        self.btn_zoom_out = QtWidgets.QPushButton(self.centralwidget)
        self.btn_zoom_out.setGeometry(QRect(500, 650, 89, 25))
        self.btn_zoom_out.setText("zoom_out")
        self.btn_zoom_out.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_zoom_out.setDisabled(True)
        self.btn_zoom_out.clicked.connect(self.set_zoom_out)

        ### mouseMove ###
        self.pmap.setMouseTracking(True)
        self.pmap.mouseMoveEvent = self.get_position
        self.label_get_pos = QtWidgets.QLabel(self)
        self.label_get_pos.setGeometry(700, 650, 190, 20)
        self.label_get_pos.setText('current position = (x,y)')
        self.label_get_pos.setStyleSheet('font-size: 12px;')

        ### mousePress ###
        self.pmap.mousePressEvent = self.get_clicked_position
        self.label_click_pos = QtWidgets.QLabel(self)
        self.label_click_pos.setGeometry(700, 670, 190, 20)
        self.label_click_pos.setText('clicked position = (x,y)')
        self.label_click_pos.setStyleSheet('font-size: 12px;')

        ### show img.shape ###
        self.label_img_shape = QtWidgets.QLabel(self)
        self.label_img_shape.setGeometry(575, 710, 500, 20)

        ### label_button ###
        self.btn_label = QtWidgets.QPushButton(self)
        self.btn_label.setText('Create RectBox')
        self.btn_label.setGeometry(980, 20, 100, 24)
        self.btn_label.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_label.setDisabled(True)
        self.btn_label.clicked.connect(self.make_label)

        ### label_list ###
        self.label_list = QtWidgets.QLabel(self)
        self.label_list.setText('Box Labels')
        self.label_list.setGeometry(980, 45, 150, 30)
        self.label_list.setStyleSheet('font-size: 12px;')

        self.hideBox = QtWidgets.QCheckBox(self)
        self.hideBox.move(980, 70)
        self.hideBox.setText('Hide Box')
        self.hideBox.clicked.connect(lambda: self.hideBbox(self.hideBox))

        self.listwidget = QtWidgets.QListWidget(self)
        self.listwidget.addItems([])
        self.listwidget.setGeometry(980, 96, 315, 140)
        self.listwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.listwidget.clicked.connect(self.showObject)
        self.listwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.listwidget.customContextMenuRequested.connect(self.on_context_menu_labimg)

        self.label_clear = QtWidgets.QPushButton(self)
        self.label_clear.setText('Delete all')
        self.label_clear.setGeometry(1235, 242, 60, 20)
        self.label_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.label_clear.clicked.connect(self.allbboxClear)

        ### paste_button ###
        self.btn_paste = QtWidgets.QPushButton(self)
        self.btn_paste.setText('Paste Image')
        self.btn_paste.setGeometry(980, 275, 100, 24)
        self.btn_paste.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_paste.setDisabled(True)
        self.btn_paste.clicked.connect(self.pasteImg)

        self.label_pasteimg = QtWidgets.QLabel(self)
        self.label_pasteimg.setText('Image')
        self.label_pasteimg.setGeometry(980, 300, 80, 30)
        self.label_pasteimg.setStyleSheet('font-size: 12px;')

        self.Hflip = QtWidgets.QCheckBox(self)
        self.Hflip.move(1020, 306)
        self.Hflip.setText('HorizontalFlip')
        self.Hflip.setDisabled(True)
        self.Hflip.clicked.connect(lambda: self.Hflippimg(self.Hflip))

        self.white_canvas = QPixmap(100, 100)
        self.white_canvas.fill(QColor('#ffffff'))
        self.pmap_pasteimg = QtWidgets.QLabel(self)
        self.pmap_pasteimg.setGeometry(980, 330, 100, 100)
        self.pmap_pasteimg.setStyleSheet('border: 1px solid #D3D3D3;')
        self.pmap_pasteimg.setPixmap(self.white_canvas)

        self.btn_chooseimg = QtWidgets.QPushButton(self)
        self.btn_chooseimg.setText('Choose')
        self.btn_chooseimg.setGeometry(1030, 436, 50, 20)
        self.btn_chooseimg.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_chooseimg.clicked.connect(self.chooseImg)

        self.btn_add = QtWidgets.QPushButton(self)
        self.btn_add.setText('Add')
        self.btn_add.setGeometry(1030, 462, 50, 20)
        self.btn_add.setStyleSheet(STYLE_BUTTON_SECONDARY_DISABLED)
        self.btn_add.setDisabled(True)
        self.btn_add.clicked.connect(self.inputPimg)

        self.btn_reset = QtWidgets.QPushButton(self)
        self.btn_reset.setText('Reset')
        self.btn_reset.setGeometry(1245, 495, 50, 20)
        self.btn_reset.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_reset.clicked.connect(self.resetVal)

        ### Paste_imgae_QListWidget ###
        self.pimg_list = QtWidgets.QLabel(self)
        self.pimg_list.setText('Paste Images')
        self.pimg_list.setGeometry(980, 510, 180, 30)
        self.pimg_list.setStyleSheet('font-size: 12px;')

        self.pimglistwidget = QtWidgets.QListWidget(self)
        self.pimglistwidget.addItems([])
        self.pimglistwidget.setGeometry(980, 542, 315, 140)
        self.pimglistwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.pimglistwidget.clicked.connect(self.showPimg)
        self.pimglistwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.pimglistwidget.customContextMenuRequested.connect(self.on_context_menu_pasteimg)

        self.pimg_clear = QtWidgets.QPushButton(self)
        self.pimg_clear.setText('Delete all')
        self.pimg_clear.setGeometry(1235, 688, 60, 20)
        self.pimg_clear.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.pimg_clear.clicked.connect(self.allpimgClear)

        self.mbox = QtWidgets.QMessageBox(self)

        ### menu_File ###
        self.menubar = QtWidgets.QMenuBar(self)
        self.menu_file = QtWidgets.QMenu('File')

        self.action_open = QAction('Open Image')
        self.action_open.setShortcut('Ctrl+o')
        self.action_open.triggered.connect(self.newFile)
        self.menu_file.addAction(self.action_open)

        self.action_input = QAction('Input Object')
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
        self.menu_file.addSeparator()

        self.action_saveimg = QAction('Save Image')
        self.action_saveimg.setDisabled(True)
        self.action_saveimg.triggered.connect(self.saveFile)
        self.menu_file.addAction(self.action_saveimg)

        self.action_savelab = QAction('Save Label')
        self.action_savelab.setDisabled(True)
        self.action_savelab.triggered.connect(self.saveLabel)
        self.menu_file.addAction(self.action_savelab)
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
        self.menubar.addMenu(self.menu_edit)

        ### open_button ###
        self.btn_open = QtWidgets.QPushButton(self)
        self.btn_open.setText('Open Image')
        self.btn_open.setGeometry(10, 20, 100, 24)
        self.btn_open.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_open.clicked.connect(self.newFile)

        self.btn_inputobj = QtWidgets.QPushButton(self)
        self.btn_inputobj.setText('Input Class')
        self.btn_inputobj.setGeometry(10, 54, 100, 24)
        self.btn_inputobj.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_inputobj.setDisabled(True)
        self.btn_inputobj.clicked.connect(self.inputObj)

        self.btn_loadlab = QtWidgets.QPushButton(self)
        self.btn_loadlab.setText('Load Label')
        self.btn_loadlab.setGeometry(10, 108, 100, 24)
        self.btn_loadlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_loadlab.setDisabled(True)
        self.btn_loadlab.clicked.connect(self.loadLabel)

        self.btn_showlab = QtWidgets.QPushButton(self)
        self.btn_showlab.setText('Show Label')
        self.btn_showlab.setGeometry(10, 162, 100, 24)
        self.btn_showlab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_showlab.setDisabled(True)
        self.btn_showlab.clicked.connect(self.showLabel)

        self.btn_saveimg = QtWidgets.QPushButton(self)
        self.btn_saveimg.setText('Save Image')
        self.btn_saveimg.setGeometry(10, 566, 100, 24)
        self.btn_saveimg.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_saveimg.setDisabled(True)
        self.btn_saveimg.clicked.connect(self.saveFile)

        self.btn_savelab = QtWidgets.QPushButton(self)
        self.btn_savelab.setText('Save Label')
        self.btn_savelab.setGeometry(10, 600, 100, 24)
        self.btn_savelab.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_savelab.setDisabled(True)
        self.btn_savelab.clicked.connect(self.saveLabel)

        self.btn_close = QtWidgets.QPushButton(self)
        self.btn_close.setText('Quit')
        self.btn_close.setGeometry(10, 684, 100, 24)
        self.btn_close.setStyleSheet(STYLE_BUTTON_PRIMARY)
        self.btn_close.clicked.connect(self.closeFile)

    def adjustUi(self):
        self.label_adj_1 = QtWidgets.QLabel(self)
        self.label_adj_1.setGeometry(1125, 330, 100, 15)
        self.label_adj_1.setText('Resize')
        self.label_adj_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_val_1 = QtWidgets.QLabel(self)
        self.label_val_1.setGeometry(1260, 350, 40, 20)
        self.label_val_1.setText("100 %")
        self.label_val_1.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider_1 = QtWidgets.QSlider(self)
        self.slider_1.setOrientation(Qt.Orientation.Horizontal)
        self.slider_1.setGeometry(1090, 350, 170, 20)
        self.slider_1.setRange(0, 100)
        self.slider_1.setValue(50)
        self.slider_1.valueChanged.connect(self.controlpimg)

        self.label_adj_2 = QtWidgets.QLabel(self)
        self.label_adj_2.setGeometry(1125, 370, 100, 15)
        self.label_adj_2.setText('Rotate')
        self.label_adj_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_val_2 = QtWidgets.QLabel(self)
        self.label_val_2.setGeometry(1260, 390, 40, 20)
        self.label_val_2.setText('0 °')
        self.label_val_2.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider_2 = QtWidgets.QSlider(self)
        self.slider_2.setOrientation(Qt.Orientation.Horizontal)
        self.slider_2.setGeometry(1090, 390, 170, 20)
        self.slider_2.setRange(0, 360)
        self.slider_2.setValue(0)
        self.slider_2.valueChanged.connect(self.controlpimg)

        self.label_adj_3 = QtWidgets.QLabel(self)
        self.label_adj_3.setGeometry(1125, 410, 100, 15)
        self.label_adj_3.setText('Brightness')
        self.label_adj_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_val_3 = QtWidgets.QLabel(self)
        self.label_val_3.setGeometry(1260, 430, 40, 20)
        self.label_val_3.setText('100')
        self.label_val_3.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider_3 = QtWidgets.QSlider(self)
        self.slider_3.setOrientation(Qt.Orientation.Horizontal)
        self.slider_3.setGeometry(1090, 430, 170, 20)
        self.slider_3.setRange(0, 200)
        self.slider_3.setValue(100)
        self.slider_3.valueChanged.connect(self.controlpimg)

        self.label_adj_4 = QtWidgets.QLabel(self)
        self.label_adj_4.setGeometry(1125, 450, 100, 15)
        self.label_adj_4.setText('Contrast')
        self.label_adj_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.label_val_4 = QtWidgets.QLabel(self)
        self.label_val_4.setGeometry(1260, 470, 40, 20)
        self.label_val_4.setText('100')
        self.label_val_4.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.slider_4 = QtWidgets.QSlider(self)
        self.slider_4.setOrientation(Qt.Orientation.Horizontal)
        self.slider_4.setGeometry(1090, 470, 170, 20)
        self.slider_4.setRange(0, 200)
        self.slider_4.setValue(100)
        self.slider_4.valueChanged.connect(self.controlpimg)

    def getslidervalue(self):
        self.set_slider_value(self.slider_zoom.value() + 1)

    def set_img_ratio(self):
        self.ratio_rate = pow(10, (self.ratio_value - 50) / 50)
        self.qpixmap_height = int(self.origin_height * self.ratio_rate)
        self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)

        if self.hideBox.isChecked() == False:
            draw_bboxes_on_canvas(self.canvas, self.data + self.pimg_data)
        draw_paste_images_on_canvas(self.canvas, self.paste_images)
        self.pmap.setPixmap(self.canvas)
        self.update()

        self.__update_img()
        self.__update_text_ratio()
        self.__update_text_img_shape()

    def __update_img(self):
        self.pmap.setPixmap(self.canvas)
        self.pmap.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop)

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
        self.pmap.mousePressEvent = self.paint

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
            self.pmap.setPixmap(self.canvas)
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
                    self.pmap.setPixmap(self.canvas)
                    self.update()
                    self.qInput()
                    self.x, self.y = None, None
                else:
                    self.x, self.y = None, None
                    if self.hideBox.isChecked():
                        self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                        self.pmap.setPixmap(self.canvas)
                        self.update()
                    else:
                        self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                        draw_bboxes_on_canvas(self.canvas, self.data + self.pimg_data)
                        self.pmap.setPixmap(self.canvas)
                        self.update()

    def qInput(self):
        item, ok = QtWidgets.QInputDialog().getItem(
            self, '', 'Enter object name', self.object_list, 0
        )
        if ok:
            self.listwidget.addItem(item)
            if item not in self.object_list:
                self.object_list.append(item)
            self.data.append([
                item, self.x, self.y, self.last_x, self.last_y,
                self.canvas.width(), self.canvas.height()
            ])
            real_x = int(self.x * self.origin_width / self.canvas.width())
            real_y = int(self.y * self.origin_height / self.canvas.height())
            real_last_x = int(self.last_x * self.origin_width / self.canvas.width())
            real_last_y = int(self.last_y * self.origin_height / self.canvas.height())
            self.real_data.append([item, real_x, real_y, real_last_x, real_last_y])
            self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')

            if self.hideBox.isChecked():
                self.hideBox.toggle()
                draw_bboxes_on_canvas(self.canvas, self.data[:-1] + self.pimg_data)
                self.pmap.setPixmap(self.canvas)
                self.update()
        else:
            if self.hideBox.isChecked():
                self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                self.pmap.setPixmap(self.canvas)
                self.update()
            else:
                self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                draw_bboxes_on_canvas(self.canvas, self.data + self.pimg_data)
                self.pmap.setPixmap(self.canvas)
                self.update()

    def hideBbox(self, cb):
        try:
            if cb.isChecked():
                self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                draw_paste_images_on_canvas(self.canvas, self.paste_images)
            else:
                draw_bboxes_on_canvas(self.canvas, self.data + self.pimg_data)
            self.pmap.setPixmap(self.canvas)
            self.update()
        except (AttributeError, ZeroDivisionError):
            return

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
        self.pmap.setPixmap(copy_canvas)
        self.update()

    def bboxClear(self):
        try:
            self.data.pop(self.ith1)
            self.real_data.pop(self.ith1)
            self.listwidget.takeItem(self.ith1)
            self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')
            self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
            draw_paste_images_on_canvas(self.canvas, self.paste_images)
            self.pmap.setPixmap(self.canvas)
            self.update()
            self.hideBox.setChecked(False)
            self.hideBbox(self.hideBox)
        except (IndexError, AttributeError):
            return

    def allbboxClear(self):
        try:
            ret = self.mbox.question(
                self, 'question', 'Delete all?',
                self.mbox.StandardButton.Cancel, self.mbox.StandardButton.Ok
            )
            if ret == self.mbox.StandardButton.Ok:
                self.data.clear()
                self.real_data.clear()
                self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')
                self.listwidget.clear()
                self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                draw_paste_images_on_canvas(self.canvas, self.paste_images)
                self.pmap.setPixmap(self.canvas)
                self.update()
                self.hideBox.setChecked(False)
                self.hideBbox(self.hideBox)
        except (AttributeError, ZeroDivisionError):
            return

    def bboxRename(self):
        try:
            text, ok = QtWidgets.QInputDialog().getItem(
                self, '', 'Enter object name', self.object_list, 0
            )
            if ok:
                self.data[self.ith1][0] = text
                self.real_data[self.ith1][0] = text
                item = self.listwidget.item(self.ith1)
                item.setText(text)
                if text not in self.object_list:
                    self.object_list.append(text)
        except (IndexError, AttributeError):
            return

    def chooseImg(self):
        filePath, filetype = QtWidgets.QFileDialog.getOpenFileName(
            self, directory='rembg_img', filter='IMAGE(*.jpg *.png *.gif *.bmp)'
        )
        if filePath:
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
                int(100 * self.pasteimg_width / self.pasteimg_height), 100
            )
        else:
            self.paste_canvas = self.paste_canvas.scaled(
                100, int(100 * self.pasteimg_height / self.pasteimg_width)
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
            self.canvas = self.pasteimg_canvas
            self.btn_add.setDisabled(True)
        else:
            self.pmap.setPixmap(self.canvas)
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
        self.pmap.setPixmap(copy_canvas)
        self.update()

    def pimgClear(self):
        try:
            self.pimg_data.pop(self.ith2)
            self.real_pimg_data.pop(self.ith2)
            self.paste_images.pop(self.ith2)
            self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
            self.pimglistwidget.takeItem(self.ith2)
            self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
            draw_paste_images_on_canvas(self.canvas, self.paste_images)
            self.pmap.setPixmap(self.canvas)
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
                self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
                self.pimglistwidget.clear()
                self.canvas = self.origin_canvas.scaledToHeight(self.qpixmap_height)
                self.pmap.setPixmap(self.canvas)
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
            self.pmap.setPixmap(self.pasteimg_canvas)
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
        self.pmap.mousePressEvent = self.paste

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
        context.exec(self.mapToGlobal(pos + QPoint(980, 96)))

    def on_context_menu_pasteimg(self, pos):
        context = QtWidgets.QMenu(self)
        self.action_pimgrename = QAction("Rename", self)
        self.action_pimgdelete = QAction("Delete", self)
        context.addAction(self.action_pimgrename)
        context.addAction(self.action_pimgdelete)
        self.action_pimgrename.triggered.connect(self.pimgRename)
        self.action_pimgdelete.triggered.connect(self.pimgClear)
        context.exec(self.mapToGlobal(pos + QPoint(980, 542)))

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

                self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')
                self.pimg_list.setText(f'Paste Images  (Total: {len(self.real_pimg_data)})')
                self.listwidget.clear()
                self.pimglistwidget.clear()
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
                self.canvas = QPixmap.fromImage(qimg)
                self.origin_canvas = self.canvas.copy()
                self.pmap.setPixmap(self.canvas)
                self.ratio_value = 50
                self.set_img_ratio()

    def inputObj(self):
        self.nw2 = InputWindow(main_widget=self, is_confirm_quit=True)
        self.nw2.show()

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
                    with open(filePath) as file_obj:
                        qpainter = QPainter()
                        qpainter.begin(self.canvas)
                        for line in file_obj:
                            line = line.strip('\n')
                            data = line.split()
                            obj = int(data[0])
                            x, y, w, h = [float(num) for num in data[1:]]
                            x1 = int(x * self.origin_width - w * self.origin_width / 2)
                            y1 = int(y * self.origin_height - h * self.origin_height / 2)
                            x2 = int(x * self.origin_width + w * self.origin_width / 2)
                            y2 = int(y * self.origin_height + h * self.origin_height / 2)
                            obj_name = self.object_list[obj]
                            self.real_data.append([obj_name, x1, y1, x2, y2])
                            self.data.append([
                                obj_name, x1, y1, x2, y2,
                                self.origin_width, self.origin_height
                            ])
                            self.listwidget.addItem(obj_name)
                            x1 *= self.canvas.width() / self.origin_width
                            y1 *= self.canvas.height() / self.origin_height
                            x2 *= self.canvas.width() / self.origin_width
                            y2 *= self.canvas.height() / self.origin_height
                            qpainter.setPen(QPen(QColor('#00ff00'), 3))
                            qpainter.drawPoint(int(x1), int(y1))
                            qpainter.setPen(QPen(QColor('#00ff00'), 3))
                            qpainter.drawPoint(int(x2), int(y2))
                            qpainter.setPen(QPen(QColor('#00ff00'), 1))
                            qpainter.drawRect(
                                int(x1), int(y1),
                                abs(int(x2 - x1)), abs(int(y2 - y1))
                            )
                        qpainter.end()
                        self.pmap.setPixmap(self.canvas)
                        self.update()

                    self.label_list.setText(f'Box Labels  (Total: {len(self.real_data)})')
                    self.hideBox.setChecked(False)
                    self.hideBbox(self.hideBox)
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
