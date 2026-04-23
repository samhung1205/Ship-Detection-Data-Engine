"""
Advanced copy-paste effects dialog.
"""
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt


class PasteEffectsDialog(QtWidgets.QDialog):
    def __init__(
        self,
        parent=None,
        *,
        shadow_enabled: bool,
        shadow_opacity_pct: int,
        shadow_offset_px: int,
        motion_blur_enabled: bool,
        motion_blur_length: int,
        motion_blur_angle_deg: int,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Paste Effects")
        self.setModal(True)
        self.resize(392, 286)

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(10)

        hint = QtWidgets.QLabel(
            "Gaussian Blur softens uniformly. Motion Blur smears in one direction to simulate movement."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(hint)

        shadow_group = QtWidgets.QGroupBox("Shadow")
        shadow_layout = QtWidgets.QGridLayout(shadow_group)
        shadow_layout.setContentsMargins(10, 10, 10, 10)
        shadow_layout.setHorizontalSpacing(8)
        shadow_layout.setVerticalSpacing(10)
        self.chk_shadow = QtWidgets.QCheckBox("Enable shadow")
        self.chk_shadow.setChecked(shadow_enabled)
        shadow_layout.addWidget(self.chk_shadow, 0, 0, 1, 3)

        shadow_layout.addWidget(QtWidgets.QLabel("Strength"), 1, 0)
        self.slider_shadow_strength = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_shadow_strength.setRange(0, 100)
        self.slider_shadow_strength.setValue(int(shadow_opacity_pct))
        self.slider_shadow_strength.setFixedHeight(32)
        shadow_layout.addWidget(self.slider_shadow_strength, 1, 1)
        self.lbl_shadow_strength = QtWidgets.QLabel()
        self.lbl_shadow_strength.setMinimumWidth(48)
        shadow_layout.addWidget(self.lbl_shadow_strength, 1, 2)

        shadow_layout.addWidget(QtWidgets.QLabel("Offset"), 2, 0)
        self.slider_shadow_offset = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_shadow_offset.setRange(0, 24)
        self.slider_shadow_offset.setValue(int(shadow_offset_px))
        self.slider_shadow_offset.setFixedHeight(32)
        shadow_layout.addWidget(self.slider_shadow_offset, 2, 1)
        self.lbl_shadow_offset = QtWidgets.QLabel()
        self.lbl_shadow_offset.setMinimumWidth(48)
        shadow_layout.addWidget(self.lbl_shadow_offset, 2, 2)
        shadow_layout.setRowMinimumHeight(1, 34)
        shadow_layout.setRowMinimumHeight(2, 34)
        layout.addWidget(shadow_group)

        motion_group = QtWidgets.QGroupBox("Motion Blur")
        motion_layout = QtWidgets.QGridLayout(motion_group)
        motion_layout.setContentsMargins(10, 10, 10, 10)
        motion_layout.setHorizontalSpacing(8)
        motion_layout.setVerticalSpacing(10)
        self.chk_motion = QtWidgets.QCheckBox("Enable motion blur")
        self.chk_motion.setChecked(motion_blur_enabled)
        motion_layout.addWidget(self.chk_motion, 0, 0, 1, 3)

        motion_layout.addWidget(QtWidgets.QLabel("Length"), 1, 0)
        self.slider_motion_length = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_motion_length.setRange(0, 25)
        self.slider_motion_length.setValue(int(motion_blur_length))
        self.slider_motion_length.setFixedHeight(32)
        motion_layout.addWidget(self.slider_motion_length, 1, 1)
        self.lbl_motion_length = QtWidgets.QLabel()
        self.lbl_motion_length.setMinimumWidth(48)
        motion_layout.addWidget(self.lbl_motion_length, 1, 2)

        motion_layout.addWidget(QtWidgets.QLabel("Angle"), 2, 0)
        self.slider_motion_angle = QtWidgets.QSlider(Qt.Orientation.Horizontal)
        self.slider_motion_angle.setRange(0, 180)
        self.slider_motion_angle.setValue(int(motion_blur_angle_deg))
        self.slider_motion_angle.setFixedHeight(32)
        motion_layout.addWidget(self.slider_motion_angle, 2, 1)
        self.lbl_motion_angle = QtWidgets.QLabel()
        self.lbl_motion_angle.setMinimumWidth(48)
        motion_layout.addWidget(self.lbl_motion_angle, 2, 2)
        motion_layout.setRowMinimumHeight(1, 34)
        motion_layout.setRowMinimumHeight(2, 34)
        layout.addWidget(motion_group)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.StandardButton.Ok
            | QtWidgets.QDialogButtonBox.StandardButton.Cancel
            | QtWidgets.QDialogButtonBox.StandardButton.Reset
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        buttons.button(QtWidgets.QDialogButtonBox.StandardButton.Reset).clicked.connect(
            self._reset_defaults
        )
        layout.addWidget(buttons)

        self.slider_shadow_strength.valueChanged.connect(self._refresh_labels)
        self.slider_shadow_offset.valueChanged.connect(self._refresh_labels)
        self.slider_motion_length.valueChanged.connect(self._refresh_labels)
        self.slider_motion_angle.valueChanged.connect(self._refresh_labels)
        self._refresh_labels()

    def values(self) -> dict[str, int | bool]:
        return {
            "shadow_enabled": self.chk_shadow.isChecked(),
            "shadow_opacity_pct": self.slider_shadow_strength.value(),
            "shadow_offset_px": self.slider_shadow_offset.value(),
            "motion_blur_enabled": self.chk_motion.isChecked(),
            "motion_blur_length": self.slider_motion_length.value(),
            "motion_blur_angle_deg": self.slider_motion_angle.value(),
        }

    def _reset_defaults(self) -> None:
        self.chk_shadow.setChecked(False)
        self.slider_shadow_strength.setValue(40)
        self.slider_shadow_offset.setValue(8)
        self.chk_motion.setChecked(False)
        self.slider_motion_length.setValue(9)
        self.slider_motion_angle.setValue(0)

    def _refresh_labels(self) -> None:
        self.lbl_shadow_strength.setText(f"{self.slider_shadow_strength.value()} %")
        self.lbl_shadow_offset.setText(f"{self.slider_shadow_offset.value()} px")
        self.lbl_motion_length.setText(str(self.slider_motion_length.value()))
        self.lbl_motion_angle.setText(f"{self.slider_motion_angle.value()} °")
