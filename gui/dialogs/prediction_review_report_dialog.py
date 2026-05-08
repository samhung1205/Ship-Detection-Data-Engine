"""
Prediction review summary dialog for folder / project workflows.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from sdde.prediction_review_report import (
    PredictionReviewReport,
    build_prediction_review_report_summary,
    export_prediction_review_report_csv,
    export_prediction_review_report_json,
)


class PredictionReviewReportDialog(QtWidgets.QDialog):
    """Modal dialog that shows folder / project review summary."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        report: PredictionReviewReport,
        scope_label: str,
        detail_label: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Prediction Review Summary")
        self.resize(980, 620)
        self._report = report
        self._scope_label = scope_label
        self._detail_label = detail_label
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        scope_lbl = QtWidgets.QLabel(f"Scope: {self._scope_label}")
        scope_lbl.setStyleSheet("color: #666; margin-bottom: 2px;")
        layout.addWidget(scope_lbl)
        if self._detail_label:
            detail = QtWidgets.QLabel(self._detail_label)
            detail.setStyleSheet("color: #666; margin-bottom: 4px;")
            detail.setWordWrap(True)
            layout.addWidget(detail)

        summary = build_prediction_review_report_summary(self._report)
        summary_lbl = QtWidgets.QLabel(
            "  |  ".join(
                [
                    f"Images: {summary['total_images']}",
                    f"With preds: {summary['images_with_predictions']}",
                    f"Reviewed: {summary['reviewed_images']}",
                    f"Partial: {summary['partial_images']}",
                    f"Pending: {summary['pending_images']}",
                    f"No preds: {summary['no_prediction_images']}",
                    f"Accepted: {summary['total_accepted_predictions']}",
                    f"Rejected: {summary['total_rejected_predictions']}",
                    f"Remaining: {summary['total_remaining_predictions']}",
                ]
            )
        )
        summary_lbl.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 6px;")
        summary_lbl.setWordWrap(True)
        layout.addWidget(summary_lbl)

        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(self._make_overview_table(summary), "Overview")
        tabs.addTab(self._make_entries_table(), "Images")
        layout.addWidget(tabs)

        btn_row = QtWidgets.QHBoxLayout()
        btn_csv = QtWidgets.QPushButton("Export CSV…")
        btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_csv)
        btn_json = QtWidgets.QPushButton("Export JSON…")
        btn_json.clicked.connect(self._export_json)
        btn_row.addWidget(btn_json)
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _make_overview_table(self, summary: dict[str, int | str]) -> QtWidgets.QTableWidget:
        rows = [
            ("Total images", summary["total_images"]),
            ("Images with predictions", summary["images_with_predictions"]),
            ("Reviewed images", summary["reviewed_images"]),
            ("Partial images", summary["partial_images"]),
            ("Pending images", summary["pending_images"]),
            ("No prediction images", summary["no_prediction_images"]),
            ("Original predictions", summary["total_original_predictions"]),
            ("Accepted predictions", summary["total_accepted_predictions"]),
            ("Rejected predictions", summary["total_rejected_predictions"]),
            ("Remaining predictions", summary["total_remaining_predictions"]),
        ]
        table = QtWidgets.QTableWidget(self)
        table.setColumnCount(2)
        table.setHorizontalHeaderLabels(["Metric", "Value"])
        table.setRowCount(len(rows))
        table.horizontalHeader().setStretchLastSection(True)
        for row, (label, value) in enumerate(rows):
            table.setItem(row, 0, _ro(str(label)))
            table.setItem(row, 1, _ro(str(value)))
        table.resizeColumnsToContents()
        return table

    def _make_entries_table(self) -> QtWidgets.QTableWidget:
        table = QtWidgets.QTableWidget(self)
        headers = [
            "Image",
            "Status",
            "Has preds",
            "Original",
            "Accepted",
            "Rejected",
            "Remaining",
            "Prediction file",
        ]
        table.setColumnCount(len(headers))
        table.setHorizontalHeaderLabels(headers)
        table.setRowCount(len(self._report.entries))
        table.horizontalHeader().setStretchLastSection(True)
        table.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, entry in enumerate(self._report.entries):
            table.setItem(row, 0, _ro(Path(entry.image_path).name))
            table.setItem(row, 1, _ro(entry.status))
            table.setItem(row, 2, _ro(str(entry.has_prediction).lower()))
            table.setItem(row, 3, _ro(str(entry.original_count)))
            table.setItem(row, 4, _ro(str(entry.accepted_count)))
            table.setItem(row, 5, _ro(str(entry.rejected_count)))
            table.setItem(row, 6, _ro(str(entry.remaining_count)))
            table.setItem(row, 7, _ro(entry.prediction_path))
        table.resizeColumnsToContents()
        return table

    def _export_csv(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save prediction review summary CSV",
            "prediction_review_summary.csv",
            "CSV (*.csv)",
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_prediction_review_report_csv(self._report), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def _export_json(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save prediction review summary JSON",
            "prediction_review_summary.json",
            "JSON (*.json)",
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_prediction_review_report_json(self._report), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))


def _ro(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item
