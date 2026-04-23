"""
ImgLab and ImgBlending - Entry point.
Run: python GUI.py
"""
import sys

from PyQt6 import QtWidgets

from gui import MyWidget


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    Form = MyWidget(is_confirm_quit=True)
    Form.show()
    sys.exit(app.exec())
