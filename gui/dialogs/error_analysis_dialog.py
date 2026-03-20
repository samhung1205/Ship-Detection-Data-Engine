"""
Error Analysis dialog — runs GT-vs-pred matching, shows results, allows
bookmark / notes editing, and exports error cases to CSV (PRD §14).
"""
from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from sdde.error_analysis import (
    ErrorCase,
    export_error_cases_csv,
    match_gt_pred,
    summarise_error_cases,
)
from sdde.prediction import PredictionRecord


class ErrorAnalysisDialog(QtWidgets.QDialog):
    """Modal dialog that runs analysis and presents error cases."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        gt_boxes: Sequence[tuple[str, float, float, float, float]],
        predictions: Sequence[PredictionRecord],
        image_id: str = "",
        iou_threshold: float = 0.5,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Error Analysis")
        self.resize(780, 520)
        self._cases = match_gt_pred(
            gt_boxes, predictions,
            iou_threshold=iou_threshold,
            image_id=image_id,
        )
        self._build_ui()

    @property
    def cases(self) -> list[ErrorCase]:
        return self._cases

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        summary = summarise_error_cases(self._cases)
        parts = [f"{k}: {v}" for k, v in sorted(summary.items())]
        lbl = QtWidgets.QLabel("  |  ".join(parts) if parts else "No cases.")
        lbl.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 4px;")
        layout.addWidget(lbl)

        self._table = QtWidgets.QTableWidget(self)
        headers = ["Type", "IoU", "GT class", "Pred class", "Conf", "Bookmark", "Notes"]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(len(self._cases))
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )
        self._table.horizontalHeader().setStretchLastSection(True)

        for row, c in enumerate(self._cases):
            self._table.setItem(row, 0, self._ro_item(c.error_type))
            self._table.setItem(row, 1, self._ro_item(f"{c.iou:.3f}"))
            self._table.setItem(row, 2, self._ro_item(c.gt_class))
            self._table.setItem(row, 3, self._ro_item(c.pred_class))
            self._table.setItem(row, 4, self._ro_item(f"{c.confidence:.3f}"))

            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            chk.setCheckState(
                Qt.CheckState.Checked if c.bookmarked else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, 5, chk)

            notes_item = QtWidgets.QTableWidgetItem(c.notes)
            self._table.setItem(row, 6, notes_item)

        self._table.resizeColumnsToContents()
        layout.addWidget(self._table)

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    @staticmethod
    def _ro_item(text: str) -> QtWidgets.QTableWidgetItem:
        item = QtWidgets.QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        return item

    def _sync_notes_and_bookmarks(self) -> None:
        for row, c in enumerate(self._cases):
            bm_item = self._table.item(row, 5)
            if bm_item is not None:
                c.bookmarked = bm_item.checkState() == Qt.CheckState.Checked
            notes_item = self._table.item(row, 6)
            if notes_item is not None:
                c.notes = notes_item.text()

    def _export_csv(self) -> None:
        self._sync_notes_and_bookmarks()
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save error cases CSV", "error_cases.csv", "CSV (*.csv)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_error_cases_csv(self._cases), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))
