"""
Per-box attribute editor (PRD §9 / spec §5.1.5-5.1.6).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from sdde.attributes import (
    CROWDED_CHOICES,
    DIFFICULTY_CHOICES,
    KEY_BLURRED,
    KEY_CROWDED,
    KEY_DIFFICULT_BACKGROUND,
    KEY_DIFFICULTY,
    KEY_HARD_SAMPLE,
    KEY_LOW_CONTRAST,
    KEY_OCCLUDED,
    KEY_SCENE,
    KEY_SIZE,
    KEY_TRUNCATED,
    SCENE_CHOICES,
    SIZE_TAG_CHOICES,
    normalize_attributes,
)


class AttributePanel(QtWidgets.QGroupBox):
    """Emits values_changed when user edits (not when load_from_dict)."""

    values_changed = pyqtSignal()

    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__("Attributes", parent)
        self._loading = False
        self._on_recalc_size: Optional[Callable[[], None]] = None

        layout = QtWidgets.QHBoxLayout(self)
        layout.setContentsMargins(10, 20, 10, 10)
        layout.setSpacing(12)

        self.core_group = QtWidgets.QGroupBox("Core", self)
        self.core_group.setObjectName("attr_core_group")
        self.core_group.setFlat(True)
        core_layout = QtWidgets.QFormLayout(self.core_group)
        core_layout.setContentsMargins(10, 16, 10, 10)
        core_layout.setHorizontalSpacing(10)
        core_layout.setVerticalSpacing(6)
        core_layout.setFieldGrowthPolicy(
            QtWidgets.QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow
        )

        self.study_group = QtWidgets.QGroupBox("Study Flags", self)
        self.study_group.setObjectName("attr_study_group")
        self.study_group.setFlat(True)
        study_layout = QtWidgets.QVBoxLayout(self.study_group)
        study_layout.setContentsMargins(10, 16, 10, 10)
        study_layout.setSpacing(6)

        self.combo_size = QtWidgets.QComboBox()
        self.combo_size.addItems(SIZE_TAG_CHOICES)
        self.combo_crowded = QtWidgets.QComboBox()
        self.combo_crowded.addItems(CROWDED_CHOICES)
        self.combo_difficulty = QtWidgets.QComboBox()
        self.combo_difficulty.addItems(DIFFICULTY_CHOICES)
        self.combo_scene = QtWidgets.QComboBox()
        self.combo_scene.addItems(SCENE_CHOICES)
        self.chk_hard_sample = QtWidgets.QCheckBox("Hard sample")
        self.chk_occluded = QtWidgets.QCheckBox("Occluded")
        self.chk_truncated = QtWidgets.QCheckBox("Truncated")
        self.chk_blurred = QtWidgets.QCheckBox("Blurred")
        self.chk_difficult_background = QtWidgets.QCheckBox("Difficult background")
        self.chk_low_contrast = QtWidgets.QCheckBox("Low contrast")

        self.btn_auto_size = QtWidgets.QPushButton("Auto size_tag")
        self.btn_auto_size.setToolTip("From bbox area (PRD §9.5, origin pixels)")

        core_layout.addRow("Size", self.combo_size)
        core_layout.addRow("", self.btn_auto_size)
        core_layout.addRow("Crowded", self.combo_crowded)
        core_layout.addRow("Difficulty", self.combo_difficulty)
        core_layout.addRow("Scene", self.combo_scene)

        for checkbox in (
            self.chk_hard_sample,
            self.chk_occluded,
            self.chk_truncated,
            self.chk_blurred,
            self.chk_difficult_background,
            self.chk_low_contrast,
        ):
            study_layout.addWidget(checkbox)
        study_layout.addStretch(1)

        layout.addWidget(self.core_group, 1)
        layout.addWidget(self.study_group, 1)

        for c in (
            self.combo_size,
            self.combo_crowded,
            self.combo_difficulty,
            self.combo_scene,
        ):
            c.currentTextChanged.connect(self._maybe_emit_changed)
        self.chk_hard_sample.toggled.connect(self._maybe_emit_changed)
        self.chk_occluded.toggled.connect(self._maybe_emit_changed)
        self.chk_truncated.toggled.connect(self._maybe_emit_changed)
        self.chk_blurred.toggled.connect(self._maybe_emit_changed)
        self.chk_difficult_background.toggled.connect(self._maybe_emit_changed)
        self.chk_low_contrast.toggled.connect(self._maybe_emit_changed)

        self.btn_auto_size.clicked.connect(self._on_auto_size_clicked)

    def set_recalc_size_callback(self, fn: Optional[Callable[[], None]]) -> None:
        self._on_recalc_size = fn

    def _on_auto_size_clicked(self) -> None:
        if self._on_recalc_size:
            self._on_recalc_size()

    def _maybe_emit_changed(self, *_args: object) -> None:
        if not self._loading:
            self.values_changed.emit()

    def set_enabled_editing(self, enabled: bool) -> None:
        self.combo_size.setEnabled(enabled)
        self.combo_crowded.setEnabled(enabled)
        self.combo_difficulty.setEnabled(enabled)
        self.chk_hard_sample.setEnabled(enabled)
        self.chk_occluded.setEnabled(enabled)
        self.chk_truncated.setEnabled(enabled)
        self.chk_blurred.setEnabled(enabled)
        self.chk_difficult_background.setEnabled(enabled)
        self.chk_low_contrast.setEnabled(enabled)
        self.combo_scene.setEnabled(enabled)
        self.btn_auto_size.setEnabled(enabled)

    def load_from_dict(self, attrs: Dict[str, str]) -> None:
        d = normalize_attributes(attrs)
        self._loading = True
        try:
            self.combo_size.setCurrentText(d[KEY_SIZE])
            self.combo_crowded.setCurrentText(d[KEY_CROWDED])
            self.combo_difficulty.setCurrentText(d[KEY_DIFFICULTY])
            self.chk_hard_sample.setChecked(d[KEY_HARD_SAMPLE] == "true")
            self.chk_occluded.setChecked(d[KEY_OCCLUDED] == "true")
            self.chk_truncated.setChecked(d[KEY_TRUNCATED] == "true")
            self.chk_blurred.setChecked(d[KEY_BLURRED] == "true")
            self.chk_difficult_background.setChecked(d[KEY_DIFFICULT_BACKGROUND] == "true")
            self.chk_low_contrast.setChecked(d[KEY_LOW_CONTRAST] == "true")
            self.combo_scene.setCurrentText(d[KEY_SCENE])
        finally:
            self._loading = False

    def to_dict(self) -> Dict[str, str]:
        return {
            KEY_SIZE: self.combo_size.currentText(),
            KEY_CROWDED: self.combo_crowded.currentText(),
            KEY_DIFFICULTY: self.combo_difficulty.currentText(),
            KEY_HARD_SAMPLE: "true" if self.chk_hard_sample.isChecked() else "false",
            KEY_OCCLUDED: "true" if self.chk_occluded.isChecked() else "false",
            KEY_TRUNCATED: "true" if self.chk_truncated.isChecked() else "false",
            KEY_BLURRED: "true" if self.chk_blurred.isChecked() else "false",
            KEY_DIFFICULT_BACKGROUND: "true" if self.chk_difficult_background.isChecked() else "false",
            KEY_LOW_CONTRAST: "true" if self.chk_low_contrast.isChecked() else "false",
            KEY_SCENE: self.combo_scene.currentText(),
        }
