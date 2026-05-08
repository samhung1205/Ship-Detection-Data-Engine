"""
Dataset validation dialog for batch QC workflows.
"""
from __future__ import annotations

from pathlib import Path

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from sdde.validation import (
    DatasetValidationResult,
    export_validation_issues_csv,
    export_validation_summary_json,
)


class ValidationDialog(QtWidgets.QDialog):
    """Modal dialog that shows dataset / project validation issues."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        result: DatasetValidationResult,
        scope_label: str,
        detail_label: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset QC")
        self.resize(1120, 620)
        self._result = result
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

        summary_lbl = QtWidgets.QLabel(self._summary_text())
        summary_lbl.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 6px;")
        summary_lbl.setWordWrap(True)
        layout.addWidget(summary_lbl)

        tabs = QtWidgets.QTabWidget(self)
        tabs.addTab(self._make_overview_table(), "Overview")
        tabs.addTab(self._make_issue_type_table(), "Issue types")
        tabs.addTab(self._make_issue_table(), "Issues")
        layout.addWidget(tabs)

        btn_row = QtWidgets.QHBoxLayout()
        btn_export = QtWidgets.QPushButton("Export CSV…")
        btn_export.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_export)
        btn_export_summary = QtWidgets.QPushButton("Export Summary JSON…")
        btn_export_summary.clicked.connect(self._export_summary_json)
        btn_row.addWidget(btn_export_summary)
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    def _summary_text(self) -> str:
        parts = [
            f"Images: {self._result.total_images}",
            f"Labels matched: {self._result.matched_labels}",
            f"Clean images: {self._result.clean_images}",
            f"Images with issues: {self._result.images_with_issues}",
        ]
        parts.append(f"Predictions matched: {self._result.matched_predictions}")
        parts.append(f"Issues: {self._result.total_issues}")
        issue_summary = "  |  ".join(
            f"{name}: {count}" for name, count in sorted(self._result.issue_type_counts.items())
        )
        if issue_summary:
            parts.append(issue_summary)
        return "  |  ".join(parts)

    def _make_overview_table(self) -> QtWidgets.QTableWidget:
        rows = [
            ("Total images", self._result.total_images),
            ("Matched labels", self._result.matched_labels),
            ("Matched predictions", self._result.matched_predictions),
            ("Clean images", self._result.clean_images),
            ("Images with issues", self._result.images_with_issues),
            ("Total issues", self._result.total_issues),
        ]
        t = QtWidgets.QTableWidget(self)
        t.setColumnCount(2)
        t.setHorizontalHeaderLabels(["Metric", "Value"])
        t.setRowCount(len(rows))
        t.horizontalHeader().setStretchLastSection(True)
        for row, (label, value) in enumerate(rows):
            t.setItem(row, 0, _ro(str(label)))
            t.setItem(row, 1, _ro(str(value)))
        t.resizeColumnsToContents()
        return t

    def _make_issue_type_table(self) -> QtWidgets.QTableWidget:
        type_counts = self._result.issue_type_counts
        rows: list[tuple[str, int, int]] = []
        for issue_type, count in sorted(type_counts.items()):
            image_count = sum(
                1 for image_path in self._result.per_image_issue_counts
                if any(
                    issue.image_path == image_path and issue.issue_type == issue_type
                    for issue in self._result.issues
                )
            )
            rows.append((issue_type, count, image_count))
        t = QtWidgets.QTableWidget(self)
        t.setColumnCount(3)
        t.setHorizontalHeaderLabels(["Issue type", "Count", "Images"])
        t.setRowCount(len(rows))
        t.horizontalHeader().setStretchLastSection(True)
        for row, (issue_type, count, image_count) in enumerate(rows):
            t.setItem(row, 0, _ro(issue_type))
            t.setItem(row, 1, _ro(str(count)))
            t.setItem(row, 2, _ro(str(image_count)))
        t.resizeColumnsToContents()
        return t

    def _make_issue_table(self) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget(self)
        headers = ["Image", "Source", "Issue type", "Line", "Detail", "File"]
        t.setColumnCount(len(headers))
        t.setHorizontalHeaderLabels(headers)
        t.setRowCount(len(self._result.issues))
        t.horizontalHeader().setStretchLastSection(True)
        t.setEditTriggers(QtWidgets.QAbstractItemView.EditTrigger.NoEditTriggers)
        for row, issue in enumerate(self._result.issues):
            t.setItem(row, 0, _ro(Path(issue.image_path).name))
            t.setItem(row, 1, _ro(issue.source))
            t.setItem(row, 2, _ro(issue.issue_type))
            t.setItem(row, 3, _ro("" if issue.line_no <= 0 else str(issue.line_no)))
            t.setItem(row, 4, _ro(issue.detail))
            t.setItem(row, 5, _ro(issue.file_path))
        t.resizeColumnsToContents()
        return t

    def _export_csv(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save dataset QC CSV",
            "dataset_qc.csv",
            "CSV (*.csv)",
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_validation_issues_csv(self._result), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))

    def _export_summary_json(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self,
            "Save dataset QC summary JSON",
            "dataset_qc_summary.json",
            "JSON (*.json)",
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_validation_summary_json(self._result), encoding="utf-8")
        except OSError as exc:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(exc))


def _ro(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item
