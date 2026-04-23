"""
Tile navigation panel widget (PRD §15).

Provides tile_size / tile_stride spin boxes, prev / next buttons, an index
label, a boundary warning summary, and a toggle to enter / exit tile view.  Emits signals so
main_window can update the canvas.
"""
from __future__ import annotations

from typing import Sequence

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt, pyqtSignal


class TilePanel(QtWidgets.QGroupBox):
    """Compact control group for tile / sliding-window navigation."""

    tile_config_changed = pyqtSignal()
    tile_index_changed = pyqtSignal(int)
    tile_step_requested = pyqtSignal(int, int)
    tile_view_toggled = pyqtSignal(bool)
    tile_overview_toggled = pyqtSignal(bool)

    def __init__(self, parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__("Tile view", parent)
        self._tile_count = 0
        self._current_index = 0
        self._step_enabled = {"up": False, "down": False, "left": False, "right": False}
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 18, 8, 8)
        layout.setSpacing(6)

        self.setStyleSheet("""
            QGroupBox {
                font-size: 12px;
                font-weight: 600;
            }
            QLabel, QCheckBox, QSpinBox, QPushButton {
                font-size: 12px;
            }
            QCheckBox {
                spacing: 4px;
            }
        """)

        form = QtWidgets.QFormLayout()
        form.setSpacing(6)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        form.setFormAlignment(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop)
        self.spin_size = QtWidgets.QSpinBox()
        self.spin_size.setRange(64, 4096)
        self.spin_size.setSingleStep(32)
        self.spin_size.setValue(640)
        self.spin_size.setFixedWidth(74)
        self.spin_size.valueChanged.connect(self._on_config_changed)
        form.addRow("Size:", self.spin_size)

        self.spin_stride = QtWidgets.QSpinBox()
        self.spin_stride.setRange(32, 4096)
        self.spin_stride.setSingleStep(32)
        self.spin_stride.setValue(480)
        self.spin_stride.setFixedWidth(74)
        self.spin_stride.valueChanged.connect(self._on_config_changed)
        form.addRow("Stride:", self.spin_stride)
        layout.addLayout(form)

        toggles = QtWidgets.QHBoxLayout()
        toggles.setSpacing(8)
        self.chk_enabled = QtWidgets.QCheckBox("On")
        self.chk_enabled.toggled.connect(self._on_toggle)
        self.chk_enabled.setMinimumWidth(42)
        toggles.addWidget(self.chk_enabled)

        self.chk_overview = QtWidgets.QCheckBox("Overview")
        self.chk_overview.toggled.connect(self.tile_overview_toggled.emit)
        self.chk_overview.setMinimumWidth(68)
        toggles.addWidget(self.chk_overview)
        toggles.addStretch(1)
        layout.addLayout(toggles)

        btn_w = 28
        btn_h = 22
        index_w = 58
        index_h = 28

        def _center_host(widget: QtWidgets.QWidget, width: int, height: int) -> QtWidgets.QWidget:
            host = QtWidgets.QWidget(self)
            host.setFixedSize(width, height)
            host_layout = QtWidgets.QHBoxLayout(host)
            host_layout.setContentsMargins(0, 0, 0, 0)
            host_layout.addStretch(1)
            host_layout.addWidget(widget)
            host_layout.addStretch(1)
            return host

        self.btn_up = QtWidgets.QPushButton("↑")
        self.btn_up.setFixedSize(btn_w, btn_h)
        self.btn_up.setStyleSheet(
            "font-size: 14px; font-weight: 600; padding: 0px; padding-top: 2px;"
        )
        self.btn_up.clicked.connect(lambda: self.tile_step_requested.emit(0, -1))
        self._btn_up_host = _center_host(self.btn_up, index_w, btn_h)

        self.btn_left = QtWidgets.QPushButton("←")
        self.btn_left.setFixedSize(btn_w, btn_h)
        self.btn_left.setStyleSheet("font-size: 14px; font-weight: 600; padding: 0px;")
        self.btn_left.clicked.connect(lambda: self.tile_step_requested.emit(-1, 0))
        self._btn_left_host = _center_host(self.btn_left, btn_w, index_h)

        self.lbl_index = QtWidgets.QLabel("0 / 0")
        self.lbl_index.setFixedSize(index_w, index_h)
        self.lbl_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lbl_index.setStyleSheet("font-size: 12px; font-weight: 600; padding: 0px;")

        self.btn_right = QtWidgets.QPushButton("→")
        self.btn_right.setFixedSize(btn_w, btn_h)
        self.btn_right.setStyleSheet("font-size: 14px; font-weight: 600; padding: 0px;")
        self.btn_right.clicked.connect(lambda: self.tile_step_requested.emit(1, 0))
        self._btn_right_host = _center_host(self.btn_right, btn_w, index_h)

        self.btn_down = QtWidgets.QPushButton("↓")
        self.btn_down.setFixedSize(btn_w, btn_h)
        self.btn_down.setStyleSheet("font-size: 14px; font-weight: 600; padding: 0px;")
        self.btn_down.clicked.connect(lambda: self.tile_step_requested.emit(0, 1))
        self._btn_down_host = _center_host(self.btn_down, index_w, btn_h)
        nav = QtWidgets.QGridLayout()
        nav.setContentsMargins(0, 0, 0, 0)
        nav.setHorizontalSpacing(6)
        nav.setVerticalSpacing(4)
        nav.setAlignment(Qt.AlignmentFlag.AlignHCenter)
        nav.setColumnMinimumWidth(0, btn_w)
        nav.setColumnMinimumWidth(1, index_w)
        nav.setColumnMinimumWidth(2, btn_w)
        nav.addWidget(self._btn_up_host, 0, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._btn_left_host, 1, 0, alignment=Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self.lbl_index, 1, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._btn_right_host, 1, 2, alignment=Qt.AlignmentFlag.AlignCenter)
        nav.addWidget(self._btn_down_host, 2, 1, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addLayout(nav)

        self.lbl_overlap = QtWidgets.QLabel("Overlap: 160 px")
        self.lbl_overlap.setStyleSheet("font-size: 11px; color: #666;")
        layout.addWidget(self.lbl_overlap)

        self.lbl_boundary = QtWidgets.QLabel("Boundary: off")
        self.lbl_boundary.setWordWrap(True)
        self.lbl_boundary.setStyleSheet("font-size: 11px; color: #666;")
        self.lbl_boundary.setToolTip("Boundary-crossing annotation hint for the current tile.")
        layout.addWidget(self.lbl_boundary)

        self._refresh_nav_state()

    def set_tile_count(self, n: int) -> None:
        self._tile_count = n
        if self._current_index >= n:
            self._current_index = max(0, n - 1)
        if n <= 0:
            self.set_overview_enabled(False)
        self._refresh_nav_state()

    def current_index(self) -> int:
        return self._current_index

    def overview_enabled(self) -> bool:
        return self.chk_overview.isChecked()

    def is_enabled(self) -> bool:
        return self.chk_enabled.isChecked()

    def tile_size(self) -> int:
        return self.spin_size.value()

    def tile_stride(self) -> int:
        return self.spin_stride.value()

    def set_current_index(self, index: int, *, emit_signal: bool = False) -> None:
        if self._tile_count <= 0:
            self._current_index = 0
            self._refresh_nav_state()
            return
        clamped = max(0, min(index, self._tile_count - 1))
        if clamped == self._current_index and not emit_signal:
            self._refresh_nav_state()
            return
        self._current_index = clamped
        self._refresh_nav_state()
        if emit_signal:
            self.tile_index_changed.emit(self._current_index)

    def set_overview_enabled(self, enabled: bool) -> None:
        if self.chk_overview.isChecked() == enabled:
            return
        self.chk_overview.setChecked(enabled)

    def set_step_enabled(self, *, up: bool, down: bool, left: bool, right: bool) -> None:
        self._step_enabled = {"up": up, "down": down, "left": left, "right": right}
        self._refresh_nav_state()

    def set_boundary_hint(
        self,
        *,
        enabled: bool,
        count: int,
        row_labels: Sequence[str] | None = None,
        tooltip: str = "",
    ) -> None:
        if not enabled:
            self.lbl_boundary.setText("Boundary: off")
            self.lbl_boundary.setStyleSheet("font-size: 10px; color: #666;")
            self.lbl_boundary.setToolTip("Enable tile view to inspect boundary-crossing annotations.")
            return
        if count <= 0:
            self.lbl_boundary.setText("Boundary: 0")
            self.lbl_boundary.setStyleSheet("font-size: 10px; color: #2e7d32;")
            self.lbl_boundary.setToolTip("No annotations cross the current tile boundary.")
            return
        rows_text = ", ".join(row_labels or ())
        body = f"Boundary: {count}"
        if rows_text:
            body += f"\nRows: {rows_text}"
        self.lbl_boundary.setText(body)
        self.lbl_boundary.setStyleSheet("font-size: 10px; color: #c25b00; font-weight: 600;")
        self.lbl_boundary.setToolTip(tooltip or f"{count} annotations cross the current tile boundary.")

    def _on_config_changed(self) -> None:
        overlap = max(0, self.spin_size.value() - self.spin_stride.value())
        self.lbl_overlap.setText(f"Overlap: {overlap} px")
        self.tile_config_changed.emit()

    def _on_toggle(self, checked: bool) -> None:
        if not checked:
            self.set_overview_enabled(False)
        self.tile_view_toggled.emit(checked)

    def _refresh_nav_state(self) -> None:
        enabled = self._tile_count > 0
        self.btn_up.setEnabled(enabled and self._step_enabled["up"])
        self.btn_down.setEnabled(enabled and self._step_enabled["down"])
        self.btn_left.setEnabled(enabled and self._step_enabled["left"])
        self.btn_right.setEnabled(enabled and self._step_enabled["right"])
        self.chk_overview.setEnabled(enabled and self.chk_enabled.isChecked())
        self.lbl_index.setText(f"{self._current_index + 1} / {self._tile_count}" if self._tile_count else "0 / 0")
