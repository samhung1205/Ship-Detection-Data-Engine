"""
Shared UI constants and styles for ImgLab and ImgBlending.
"""

STYLE_BUTTON_PRIMARY = '''
    QPushButton {
        font-size: 12px;
        color: #000;
        background: #E0E0E0;
        border: 1px solid #000;
        border-radius: 5px;
    }
    QPushButton:hover {
        color: #fff;
        background: #C4E1FF;
    }
    QPushButton:pressed {
        color: #000;
        background: #a9a9a9;
    }
    QPushButton:disabled {
        color:#fff;
        background:#ccc;
        border: 1px solid #aaa;
    }
'''

STYLE_BUTTON_SECONDARY = '''
    QPushButton {
        font-size: 12px;
        color: #000;
        background: #f5f5f5;
        border: 1px solid #c0c0c0;
        border-radius: 5px;
    }
    QPushButton:hover {
        color: #000;
        background: #C4E1FF;
    }
    QPushButton:pressed {
        color: #000;
        background: #a9a9a9;
    }
'''

STYLE_BUTTON_SECONDARY_DISABLED = STYLE_BUTTON_SECONDARY + '''
    QPushButton:disabled {
        color:#fff;
        background:#ccc;
        border: 1px solid #aaa;
    }
'''

STYLE_LIST_WIDGET = '''
    QListWidget::item{
        font-size:20px;
    }
    QListWidget::item:pressed{
        color:#fff;
        background:#C4E1FF;
    }
'''
