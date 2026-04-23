"""Layout smoke tests for the grouped attribute panel."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.attribute_panel import AttributePanel  # noqa: E402


APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_attribute_panel_shows_core_and_study_groups() -> None:
    panel = AttributePanel()
    panel.show()
    APP.processEvents()

    core_group = panel.findChild(QtWidgets.QGroupBox, "attr_core_group")
    study_group = panel.findChild(QtWidgets.QGroupBox, "attr_study_group")

    assert core_group is not None
    assert study_group is not None
    assert core_group.isVisible() is True
    assert study_group.isVisible() is True
    assert panel.chk_hard_sample.text() == "Hard sample"
    assert panel.chk_difficult_background.text() == "Difficult background"
    assert panel.chk_low_contrast.text() == "Low contrast"
    assert panel.btn_auto_size.isVisible() is True

    panel.close()


def test_attribute_panel_round_trip_keeps_all_values() -> None:
    panel = AttributePanel()
    attrs = {
        "size_tag": "large",
        "crowded": "true",
        "difficulty_tag": "hard",
        "hard_sample": "true",
        "occluded": "true",
        "truncated": "false",
        "blurred": "true",
        "difficult_background": "true",
        "low_contrast": "false",
        "scene_tag": "near_shore",
    }

    panel.load_from_dict(attrs)

    assert panel.to_dict() == attrs
