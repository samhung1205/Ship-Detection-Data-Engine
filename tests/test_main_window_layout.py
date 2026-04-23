"""Smoke tests for main-window right-column layout spacing."""

import os
import sys
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402
from PyQt6.QtGui import QColor, QPixmap  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.main_window import MyWidget  # noqa: E402


APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_main_window_right_column_has_clear_vertical_spacing() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    assert widget.width() >= 1540
    assert widget.height() >= 930
    assert widget.listwidget.geometry().bottom() < widget._attr_panel.geometry().top()
    assert widget._attr_panel.geometry().bottom() < widget.btn_paste.geometry().top()
    assert not widget.label_clear.geometry().intersects(widget._attr_panel.geometry())
    assert not widget.label_clear.geometry().intersects(widget.btn_paste.geometry())
    assert widget.listwidget.geometry().right() <= widget.width()
    assert widget._attr_panel.geometry().right() <= widget.width()
    assert widget.pimglistwidget.geometry().right() <= widget.width()
    assert widget.pimg_clear.geometry().right() <= widget.width()
    assert widget.pimg_clear.geometry().bottom() <= widget.height()
    assert widget._tile_panel.geometry().bottom() < widget.btn_saveimg.geometry().top()
    assert widget.btn_runmodel.geometry().bottom() < widget.pred_listwidget.geometry().top()

    widget.close()


def test_main_window_paste_controls_reset_and_stack_cleanly() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    assert hasattr(widget, "Vflip")
    assert hasattr(widget, "slider_5")
    assert hasattr(widget, "slider_6")
    assert hasattr(widget, "slider_7")
    assert widget.combo_paste_mode.currentText() == "Manual"
    assert widget.combo_paste_size.currentText() == "medium"
    assert widget.btn_paste_effects.text() == "Effects (0)"

    widget.Hflip.setChecked(True)
    widget.Vflip.setChecked(True)
    widget.slider_1.setValue(60)
    widget.slider_5.setValue(4)
    widget.slider_6.setValue(40)
    widget.slider_7.setValue(3)
    widget.resetVal()

    assert widget.Hflip.isChecked() is False
    assert widget.Vflip.isChecked() is False
    assert widget.slider_1.value() == 50
    assert widget.slider_5.value() == 0
    assert widget.slider_6.value() == 100
    assert widget.slider_7.value() == 0
    assert widget.pmap_pasteimg.geometry().width() >= 100
    assert widget.pmap_pasteimg.geometry().height() >= 120
    assert widget.btn_paste_effects.geometry().top() > widget.pmap_pasteimg.geometry().bottom()
    assert widget.btn_paste_effects.geometry().bottom() < widget.btn_chooseimg.geometry().top()
    assert widget.btn_paste_effects.geometry().left() == widget.pmap_pasteimg.geometry().left()
    assert widget.btn_paste.geometry().bottom() < widget.label_pasteimg.geometry().top()
    assert widget.combo_paste_mode.geometry().right() < widget.combo_paste_size.geometry().left()
    assert widget.lbl_paste_status.geometry().bottom() < widget.label_pasteimg.geometry().top()
    assert widget.btn_chooseimg.geometry().bottom() < widget.pimg_list.geometry().top()
    assert widget.pimglistwidget.geometry().bottom() < widget.pimg_clear.geometry().top()

    widget.close()


def test_tile_panel_content_fits_and_nav_is_centered() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    panel = widget._tile_panel
    assert panel.geometry().right() < widget.verticalLayoutWidget.geometry().left()
    assert panel.chk_overview.geometry().right() <= panel.width() - 8
    assert panel.lbl_overlap.geometry().right() <= panel.width() - 8
    assert panel.lbl_boundary.geometry().right() <= panel.width() - 8
    assert panel.btn_up.width() == 28
    assert panel.btn_up.height() == 22
    assert panel.btn_down.width() == 28
    assert panel.btn_down.height() == 22
    assert panel.btn_left.width() == 28
    assert panel.btn_left.height() == 22
    assert panel.btn_right.width() == 28
    assert panel.btn_right.height() == 22
    up_center = panel.btn_up.mapTo(panel, panel.btn_up.rect().center())
    down_center = panel.btn_down.mapTo(panel, panel.btn_down.rect().center())
    left_center = panel.btn_left.mapTo(panel, panel.btn_left.rect().center())
    right_center = panel.btn_right.mapTo(panel, panel.btn_right.rect().center())
    index_center = panel.lbl_index.mapTo(panel, panel.lbl_index.rect().center())
    assert abs(up_center.x() - index_center.x()) <= 1
    assert abs(down_center.x() - index_center.x()) <= 1
    assert abs(left_center.y() - index_center.y()) <= 1
    assert abs(right_center.y() - index_center.y()) <= 1

    widget.close()


def test_prediction_header_slider_aligns_cleanly_and_keeps_top_gap() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    slider_geo = widget.slider_pred_conf.geometry()
    label_geo = widget.lbl_pred_conf.geometry()
    value_geo = widget.lbl_pred_conf_val.geometry()

    assert slider_geo.height() >= 28
    assert abs(label_geo.center().y() - slider_geo.center().y()) <= 2
    assert abs(value_geo.center().y() - slider_geo.center().y()) <= 2
    assert slider_geo.bottom() < widget.listwidget.geometry().top()

    widget.close()


def test_build_export_image_pixmap_uses_origin_resolution_payload() -> None:
    widget = MyWidget(is_confirm_quit=False)
    widget.show()
    APP.processEvents()

    origin = QPixmap(100, 50)
    origin.fill(QColor("white"))
    widget.origin_canvas = origin
    widget.paste_images = [["display", 0.1, 0.1, 0.2, 0.2, "export", 0.3, 0.2, 0.4, 0.5]]

    with patch("gui.main_window.draw_paste_images_on_canvas") as draw:
        export_pm = widget.build_export_image_pixmap()

    assert export_pm is not None
    assert export_pm.width() == 100
    assert export_pm.height() == 50
    draw.assert_called_once()
    assert draw.call_args.kwargs["prefer_export_geometry"] is True

    widget.close()
