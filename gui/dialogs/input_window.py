"""
Input class dialog for managing object names.
"""
from PyQt6 import QtWidgets
from PyQt6.QtGui import QAction
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QCloseEvent

from ..constants import STYLE_BUTTON_SECONDARY, STYLE_LIST_WIDGET


class InputWindow(QtWidgets.QWidget):
    def __init__(self, main_widget: 'MyWidget', is_confirm_quit: bool = True):
        super().__init__()
        self.setWindowTitle('Input Class')
        self.resize(300, 240)
        self.main_widget = main_widget
        self.is_confirm_quit = is_confirm_quit
        self.object = main_widget.object_list.copy()

        main_widget.btn_label.setDisabled(True)
        main_widget.btn_paste.setDisabled(True)
        main_widget.action_load.setDisabled(True)
        main_widget.action_label.setDisabled(True)
        main_widget.action_paste.setDisabled(True)
        main_widget.btn_loadlab.setDisabled(True)
        main_widget.btn_showlab.setDisabled(True)
        main_widget.action_show.setDisabled(True)

        self.ui()

    def ui(self):
        self.label_format = QtWidgets.QLabel(self)
        self.label_format.setGeometry(10, 20, 100, 20)
        self.label_format.setText('Object name: ')

        self.input = QtWidgets.QLineEdit(self)
        self.input.setGeometry(105, 20, 100, 20)

        self.btn_input = QtWidgets.QPushButton(self)
        self.btn_input.setText('Enter')
        self.btn_input.setGeometry(230, 20, 60, 20)
        self.btn_input.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_input.clicked.connect(self.addObject)

        self.listwidget = QtWidgets.QListWidget(self)
        self.listwidget.addItems(self.object)
        self.listwidget.setGeometry(10, 50, 200, 120)
        self.listwidget.setStyleSheet(STYLE_LIST_WIDGET)
        self.listwidget.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.listwidget.customContextMenuRequested.connect(self.on_context_menu)

        self.btn_delete = QtWidgets.QPushButton(self)
        self.btn_delete.setText('Delete all')
        self.btn_delete.setGeometry(230, 120, 60, 20)
        self.btn_delete.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_delete.clicked.connect(self.clearAll)

        self.btn_save = QtWidgets.QPushButton(self)
        self.btn_save.setText('Save')
        self.btn_save.setGeometry(230, 150, 60, 20)
        self.btn_save.setStyleSheet(STYLE_BUTTON_SECONDARY)
        self.btn_save.clicked.connect(self.save_yaml)

        self.btn_ok = QtWidgets.QPushButton(self)
        self.btn_ok.setText('OK')
        self.btn_ok.setGeometry(155, 190, 90, 30)
        self.btn_ok.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #FFF;
                background: #0066FF;
                border-radius: 5px;
            }
            QPushButton:pressed {
                color: #000;
                background: #a9a9a9;
            }
        ''')
        self.btn_ok.clicked.connect(self.saveObjname)

        self.btn_cancel = QtWidgets.QPushButton(self)
        self.btn_cancel.setText('Cancel')
        self.btn_cancel.setGeometry(55, 190, 90, 30)
        self.btn_cancel.setStyleSheet('''
            QPushButton {
                font-size: 14px;
                color: #000;
                background: #E0E0E0;
                border-radius: 5px;
            }
            QPushButton:pressed {
                color: #000;
                background: #a9a9a9;
            }
        ''')
        self.btn_cancel.clicked.connect(self.closeWindow)

    def addObject(self):
        item = self.input.text()
        if item not in self.object and len(item) != 0:
            self.object.append(item)
            self.listwidget.addItem(item)
        self.input.setText('')

    def renameObject(self):
        try:
            name, ok = QtWidgets.QInputDialog.getText(self, '', 'Enter object name')
            if ok and len(name) != 0:
                if name not in self.object:
                    num = self.listwidget.currentIndex().row()
                    item = self.listwidget.item(num)
                    item.setText(name)
                    self.object[num] = name
                else:
                    mbox = QtWidgets.QMessageBox(self)
                    mbox.setText('Object name already exists!')
                    mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
                    mbox.exec()
        except (IndexError, AttributeError):
            pass

    def deleteObject(self):
        try:
            num = self.listwidget.currentIndex().row()
            self.listwidget.takeItem(num)
            self.object.pop(num)
        except (IndexError, AttributeError):
            pass

    def saveObjname(self):
        if len(self.object) == 0:
            mbox = QtWidgets.QMessageBox(self)
            mbox.setText('Not enter any object name!')
            mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            mbox.exec()
        else:
            self.main_widget.object_list = self.object.copy()
            self._enable_main_buttons()
            self.close()

    def _enable_main_buttons(self):
        m = self.main_widget
        m.btn_label.setDisabled(False)
        m.btn_paste.setDisabled(False)
        m.action_load.setDisabled(False)
        m.action_label.setDisabled(False)
        m.action_paste.setDisabled(False)
        m.btn_loadlab.setDisabled(False)
        m.btn_showlab.setDisabled(False)
        m.action_show.setDisabled(False)

    def closeWindow(self):
        if len(self.main_widget.object_list) == 0:
            mbox = QtWidgets.QMessageBox(self)
            mbox.setText('Not save any object name!')
            mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            mbox.exec()
        else:
            self._enable_main_buttons()
            self.close()

    def closeEvent(self, event: QCloseEvent) -> None:
        if len(self.main_widget.object_list) == 0:
            mbox = QtWidgets.QMessageBox(self)
            mbox.setText('Not save any object name!')
            mbox.setIcon(QtWidgets.QMessageBox.Icon.Critical)
            mbox.exec()
            event.ignore()
        else:
            self._enable_main_buttons()
            event.accept()

    def on_context_menu(self, pos):
        context = QtWidgets.QMenu(self)
        self.action_rename = QAction("Rename", self)
        self.action_delete = QAction("Delete", self)

        context.addAction(self.action_rename)
        context.addAction(self.action_delete)

        self.action_rename.triggered.connect(self.renameObject)
        self.action_delete.triggered.connect(self.deleteObject)

        context.exec(self.mapToGlobal(pos))

    def clearAll(self):
        ret = QtWidgets.QMessageBox.question(
            self, 'question', 'Delete all class name?',
            QtWidgets.QMessageBox.StandardButton.Cancel,
            QtWidgets.QMessageBox.StandardButton.Ok
        )
        if ret == QtWidgets.QMessageBox.StandardButton.Ok:
            self.object.clear()
            self.listwidget.clear()
            self.main_widget.object_list.clear()

    def save_yaml(self):
        filePath, filterType = QtWidgets.QFileDialog.getSaveFileName(
            self, directory='data.yaml', filter='YAML(*.yaml)'
        )
        if filePath:
            with open(filePath, 'w') as file:
                file.write('train: ' + '\n')
                file.write('val: ' + '\n')
                file.write('\n')
                file.write('nc: %d \n' % len(self.object))
                file.write('name: ' + str(self.object))
