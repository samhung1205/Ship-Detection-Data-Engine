"""
Per-box attribute editor (PRD §9): size_tag, crowded, difficulty_tag, scene_tag.
"""
from __future__ import annotations

from typing import Callable, Dict, Optional

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal

from sdde.attributes import (
    CROWDED_CHOICES,
    DIFFICULTY_CHOICES,
    KEY_CROWDED,
    KEY_DIFFICULTY,
    KEY_SCENE,
    KEY_SIZE,
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

        layout = QtWidgets.QFormLayout(self)
        self.combo_size = QtWidgets.QComboBox()
        self.combo_size.addItems(SIZE_TAG_CHOICES)
        self.combo_crowded = QtWidgets.QComboBox()
        self.combo_crowded.addItems(CROWDED_CHOICES)
        self.combo_difficulty = QtWidgets.QComboBox()
        self.combo_difficulty.addItems(DIFFICULTY_CHOICES)
        self.combo_scene = QtWidgets.QComboBox()
        self.combo_scene.addItems(SCENE_CHOICES)

        self.btn_auto_size = QtWidgets.QPushButton("Auto size_tag")
        self.btn_auto_size.setToolTip("From bbox area (PRD §9.5, origin pixels)")

        layout.addRow("size_tag", self.combo_size)
        layout.addRow(self.btn_auto_size)
        layout.addRow("crowded", self.combo_crowded)
        layout.addRow("difficulty_tag", self.combo_difficulty)
        layout.addRow("scene_tag", self.combo_scene)

        for c in (
            self.combo_size,
            self.combo_crowded,
            self.combo_difficulty,
            self.combo_scene,
        ):
            c.currentTextChanged.connect(self._maybe_emit_changed)

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
        self.combo_scene.setEnabled(enabled)
        self.btn_auto_size.setEnabled(enabled)

    def load_from_dict(self, attrs: Dict[str, str]) -> None:
        d = normalize_attributes(attrs)
        self._loading = True
        try:
            self.combo_size.setCurrentText(d[KEY_SIZE])
            self.combo_crowded.setCurrentText(d[KEY_CROWDED])
            self.combo_difficulty.setCurrentText(d[KEY_DIFFICULTY])
            self.combo_scene.setCurrentText(d[KEY_SCENE])
        finally:
            self._loading = False

    def to_dict(self) -> Dict[str, str]:
        return {
            KEY_SIZE: self.combo_size.currentText(),
            KEY_CROWDED: self.combo_crowded.currentText(),
            KEY_DIFFICULTY: self.combo_difficulty.currentText(),
            KEY_SCENE: self.combo_scene.currentText(),
        }
