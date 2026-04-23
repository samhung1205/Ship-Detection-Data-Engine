"""Layout smoke tests for the paste effects dialog."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.dialogs import PasteEffectsDialog  # noqa: E402


APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_paste_effects_dialog_sliders_have_headroom() -> None:
    dialog = PasteEffectsDialog(
        shadow_enabled=False,
        shadow_opacity_pct=40,
        shadow_offset_px=8,
        motion_blur_enabled=False,
        motion_blur_length=9,
        motion_blur_angle_deg=0,
    )
    dialog.show()
    APP.processEvents()

    assert dialog.slider_shadow_strength.height() >= 32
    assert dialog.slider_shadow_offset.height() >= 32
    assert dialog.slider_motion_length.height() >= 32
    assert dialog.slider_motion_angle.height() >= 32

    dialog.close()
