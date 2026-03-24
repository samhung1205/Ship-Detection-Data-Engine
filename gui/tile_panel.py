"""
Tile navigation panel widget (PRD §15).

Provides tile_size / tile_stride spin boxes, prev / next buttons, an index
label, and a toggle to enter / exit tile view.  Emits signals so
main_window can update the canvas.
"""
from __future__ import annotations

from PyQt6 import QtWidgets
from PyQt6.QtCore import pyqtSignal


class TilePanel(QtWidgets.QGroupBox):
    """Compact control group for tile / sliding-window navigation."""

    tile_config_changed = pyqtSignal()
    tile_index_changed = pyqtSignal(int)
    tile_view_toggled = pyqtSignal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Tile view", parent)
        self._tile_count = 0
        self._current_index = 0
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(4, 14, 4, 4)
        layout.setSpacing(3)

        form = QtWidgets.QFormLayout()
        form.setSpacing(2)
        self.spin_size = QtWidgets.QSpinBox()
        self.spin_size.setRange(64, 4096)
        self.spin_size.setSingleStep(32)
        self.spin_size.setValue(640)
        self.spin_size.valueChanged.connect(self._on_config_changed)
        form.addRow("Size:", self.spin_size)

        self.spin_stride = QtWidgets.QSpinBox()
        self.spin_stride.setRange(32, 4096)
        self.spin_stride.setSingleStep(32)
        self.spin_stride.setValue(480)
        self.spin_stride.valueChanged.connect(self._on_config_changed)
        form.addRow("Stride:", self.spin_stride)
        layout.addLayout(form)

        self.chk_enabled = QtWidgets.QCheckBox("On")
        self.chk_enabled.toggled.connect(self._on_toggle)
        layout.addWidget(self.chk_enabled)

        row2 = QtWidgets.QHBoxLayout()
        self.btn_prev = QtWidgets.QPushButton("<")
        self.btn_prev.setFixedWidth(24)
        self.btn_prev.clicked.connect(self._go_prev)
        row2.addWidget(self.btn_prev)

        self.lbl_index = QtWidgets.QLabel("0 / 0")
        self.lbl_index.setFixedWidth(48)
        row2.addWidget(self.lbl_index)

        self.btn_next = QtWidgets.QPushButton(">")
        self.btn_next.setFixedWidth(24)
        self.btn_next.clicked.connect(self._go_next)
        row2.addWidget(self.btn_next)
        layout.addLayout(row2)

        self.lbl_overlap = QtWidgets.QLabel("Overlap: 160 px")
        self.lbl_overlap.setStyleSheet("font-size: 10px; color: #666;")
        layout.addWidget(self.lbl_overlap)

        self._refresh_nav_state()

    def set_tile_count(self, n: int) -> None:
        self._tile_count = n
        if self._current_index >= n:
            self._current_index = max(0, n - 1)
        self._refresh_nav_state()

    def current_index(self) -> int:
        return self._current_index

    def is_enabled(self) -> bool:
        return self.chk_enabled.isChecked()

    def tile_size(self) -> int:
        return self.spin_size.value()

    def tile_stride(self) -> int:
        return self.spin_stride.value()

    def _on_config_changed(self) -> None:
        overlap = max(0, self.spin_size.value() - self.spin_stride.value())
        self.lbl_overlap.setText(f"Overlap: {overlap} px")
        self.tile_config_changed.emit()

    def _on_toggle(self, checked: bool) -> None:
        self.tile_view_toggled.emit(checked)

    def _go_prev(self) -> None:
        if self._current_index > 0:
            self._current_index -= 1
            self._refresh_nav_state()
            self.tile_index_changed.emit(self._current_index)

    def _go_next(self) -> None:
        if self._current_index < self._tile_count - 1:
            self._current_index += 1
            self._refresh_nav_state()
            self.tile_index_changed.emit(self._current_index)

    def _refresh_nav_state(self) -> None:
        self.btn_prev.setEnabled(self._current_index > 0)
        self.btn_next.setEnabled(self._current_index < self._tile_count - 1)
        self.lbl_index.setText(f"{self._current_index + 1} / {self._tile_count}" if self._tile_count else "0 / 0")
