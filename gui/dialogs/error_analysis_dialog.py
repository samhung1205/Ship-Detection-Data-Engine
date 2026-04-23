"""
Error Analysis dialog — runs GT-vs-pred matching, shows results, allows
bookmark / notes editing, and exports error cases to CSV (PRD §14).
"""
from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from sdde.error_analysis import (
    ALL_ERROR_TYPES,
    ERROR_FILTER_ALL,
    ErrorCase,
    export_error_cases_csv,
    filter_error_cases,
    gt_attributes_for_case,
    match_gt_pred,
    summarise_error_cases,
)
from sdde.prediction import PredictionRecord
from sdde.attributes import CROWDED_CHOICES, DIFFICULTY_CHOICES, SCENE_CHOICES, SIZE_TAG_CHOICES


class ErrorAnalysisDialog(QtWidgets.QDialog):
    """Modal dialog that runs analysis and presents error cases."""

    _COL_IMAGE = 0
    _COL_TYPE = 1
    _COL_IOU = 2
    _COL_GT_CLASS = 3
    _COL_PRED_CLASS = 4
    _COL_SIZE = 5
    _COL_SCENE = 6
    _COL_DIFFICULTY = 7
    _COL_CROWDED = 8
    _COL_HARD_SAMPLE = 9
    _COL_OCCLUDED = 10
    _COL_TRUNCATED = 11
    _COL_BLURRED = 12
    _COL_CONF = 13
    _COL_BOOKMARK = 14
    _COL_NOTES = 15

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        gt_boxes: Sequence[tuple[str, float, float, float, float]],
        gt_attributes: Sequence[Mapping[str, str]] | None = None,
        predictions: Sequence[PredictionRecord],
        image_id: str = "",
        iou_threshold: float = 0.5,
        cases: Sequence[ErrorCase] | None = None,
        scope_label: str = "Current image",
        detail_label: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Error Analysis")
        self.resize(1240, 620)
        self._scope_label = scope_label
        self._detail_label = detail_label
        self._cases = list(cases) if cases is not None else match_gt_pred(
            gt_boxes,
            predictions,
            gt_attributes=gt_attributes,
            iou_threshold=iou_threshold,
            image_id=image_id,
        )
        self._gt_attributes = list(gt_attributes) if gt_attributes is not None else None
        self._visible_cases: list[ErrorCase] = list(self._cases)
        self._build_ui()

    @property
    def cases(self) -> list[ErrorCase]:
        return self._cases

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        self._scope_label_widget = QtWidgets.QLabel(f"Scope: {self._scope_label}")
        self._scope_label_widget.setStyleSheet("color: #666; margin-bottom: 2px;")
        layout.addWidget(self._scope_label_widget)

        if self._detail_label:
            detail = QtWidgets.QLabel(self._detail_label)
            detail.setStyleSheet("color: #666; margin-bottom: 4px;")
            layout.addWidget(detail)

        self._summary_label = QtWidgets.QLabel("No cases.")
        self._summary_label.setStyleSheet("font-weight: bold; font-size: 13px; margin-bottom: 4px;")
        layout.addWidget(self._summary_label)

        filter_row = QtWidgets.QHBoxLayout()
        filter_row.addWidget(QtWidgets.QLabel("Type:"))
        self._filter_combo = QtWidgets.QComboBox(self)
        self._filter_combo.addItem(ERROR_FILTER_ALL)
        for error_type in ALL_ERROR_TYPES:
            if any(c.error_type == error_type for c in self._cases):
                self._filter_combo.addItem(error_type)
        self._filter_combo.currentTextChanged.connect(self._on_filter_changed)
        filter_row.addWidget(self._filter_combo)

        self._bookmarked_only = QtWidgets.QCheckBox("Bookmarked only", self)
        self._bookmarked_only.toggled.connect(self._on_filter_changed)
        filter_row.addWidget(self._bookmarked_only)
        filter_row.addStretch()
        layout.addLayout(filter_row)

        attr_row = QtWidgets.QHBoxLayout()
        self._size_combo = self._make_attr_filter_combo("Size:", SIZE_TAG_CHOICES, attr_row)
        self._scene_combo = self._make_attr_filter_combo("Scene:", SCENE_CHOICES, attr_row)
        self._difficulty_combo = self._make_attr_filter_combo("Difficulty:", DIFFICULTY_CHOICES, attr_row)
        self._crowded_combo = self._make_attr_filter_combo("Crowded:", CROWDED_CHOICES, attr_row)
        attr_row.addStretch()
        layout.addLayout(attr_row)

        attr_row_2 = QtWidgets.QHBoxLayout()
        self._hard_sample_combo = self._make_attr_filter_combo("Hard:", CROWDED_CHOICES, attr_row_2)
        self._occluded_combo = self._make_attr_filter_combo("Occluded:", CROWDED_CHOICES, attr_row_2)
        self._truncated_combo = self._make_attr_filter_combo("Truncated:", CROWDED_CHOICES, attr_row_2)
        self._blurred_combo = self._make_attr_filter_combo("Blurred:", CROWDED_CHOICES, attr_row_2)
        attr_row_2.addStretch()
        layout.addLayout(attr_row_2)

        self._table = QtWidgets.QTableWidget(self)
        headers = [
            "Image",
            "Type",
            "IoU",
            "GT class",
            "Pred class",
            "Size",
            "Scene",
            "Difficulty",
            "Crowded",
            "Hard",
            "Occ",
            "Trunc",
            "Blur",
            "Conf",
            "Bookmark",
            "Notes",
        ]
        self._table.setColumnCount(len(headers))
        self._table.setHorizontalHeaderLabels(headers)
        self._table.setRowCount(0)
        self._table.setEditTriggers(
            QtWidgets.QAbstractItemView.EditTrigger.DoubleClicked
        )
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self._table)
        self._apply_filters()

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

    def _make_attr_filter_combo(
        self,
        label: str,
        values: Sequence[str],
        row: QtWidgets.QHBoxLayout,
    ) -> QtWidgets.QComboBox:
        row.addWidget(QtWidgets.QLabel(label))
        combo = QtWidgets.QComboBox(self)
        combo.addItem(ERROR_FILTER_ALL)
        combo.addItems(list(values))
        combo.currentTextChanged.connect(self._on_filter_changed)
        row.addWidget(combo)
        return combo

    def _sync_notes_and_bookmarks(self) -> None:
        for row, c in enumerate(self._visible_cases):
            bm_item = self._table.item(row, self._COL_BOOKMARK)
            if bm_item is not None:
                c.bookmarked = bm_item.checkState() == Qt.CheckState.Checked
            notes_item = self._table.item(row, self._COL_NOTES)
            if notes_item is not None:
                c.notes = notes_item.text()

    def _on_filter_changed(self, *_args) -> None:
        self._apply_filters()

    def _apply_filters(self) -> None:
        self._sync_notes_and_bookmarks()
        self._visible_cases = filter_error_cases(
            self._cases,
            error_type=self._filter_combo.currentText(),
            bookmarked_only=self._bookmarked_only.isChecked(),
            gt_attributes=self._gt_attributes,
            size_tag=self._size_combo.currentText(),
            scene_tag=self._scene_combo.currentText(),
            difficulty_tag=self._difficulty_combo.currentText(),
            crowded=self._crowded_combo.currentText(),
            hard_sample=self._hard_sample_combo.currentText(),
            occluded=self._occluded_combo.currentText(),
            truncated=self._truncated_combo.currentText(),
            blurred=self._blurred_combo.currentText(),
        )
        self._refresh_summary_label()
        self._rebuild_table()

    def _refresh_summary_label(self) -> None:
        summary = summarise_error_cases(self._visible_cases)
        parts = [f"{k}: {summary[k]}" for k in ALL_ERROR_TYPES if summary.get(k)]
        prefix = f"Showing {len(self._visible_cases)} / {len(self._cases)}"
        self._summary_label.setText(
            prefix + ("  |  " + "  |  ".join(parts) if parts else "  |  No cases in current filter.")
        )

    def _rebuild_table(self) -> None:
        self._table.setRowCount(len(self._visible_cases))
        for row, c in enumerate(self._visible_cases):
            attrs = gt_attributes_for_case(c, self._gt_attributes)
            size_tag = attrs.get("size_tag", "-") if attrs else "-"
            scene_tag = attrs.get("scene_tag", "-") if attrs else "-"
            difficulty_tag = attrs.get("difficulty_tag", "-") if attrs else "-"
            crowded = attrs.get("crowded", "-") if attrs else "-"
            hard_sample = attrs.get("hard_sample", "-") if attrs else "-"
            occluded = attrs.get("occluded", "-") if attrs else "-"
            truncated = attrs.get("truncated", "-") if attrs else "-"
            blurred = attrs.get("blurred", "-") if attrs else "-"
            image_name = Path(c.image_id).name if c.image_id else "-"
            self._table.setItem(row, self._COL_IMAGE, self._ro_item(image_name))
            self._table.setItem(row, self._COL_TYPE, self._ro_item(c.error_type))
            self._table.setItem(row, self._COL_IOU, self._ro_item(f"{c.iou:.3f}"))
            self._table.setItem(row, self._COL_GT_CLASS, self._ro_item(c.gt_class))
            self._table.setItem(row, self._COL_PRED_CLASS, self._ro_item(c.pred_class))
            self._table.setItem(row, self._COL_SIZE, self._ro_item(size_tag))
            self._table.setItem(row, self._COL_SCENE, self._ro_item(scene_tag))
            self._table.setItem(row, self._COL_DIFFICULTY, self._ro_item(difficulty_tag))
            self._table.setItem(row, self._COL_CROWDED, self._ro_item(crowded))
            self._table.setItem(row, self._COL_HARD_SAMPLE, self._ro_item(hard_sample))
            self._table.setItem(row, self._COL_OCCLUDED, self._ro_item(occluded))
            self._table.setItem(row, self._COL_TRUNCATED, self._ro_item(truncated))
            self._table.setItem(row, self._COL_BLURRED, self._ro_item(blurred))
            self._table.setItem(row, self._COL_CONF, self._ro_item(f"{c.confidence:.3f}"))

            chk = QtWidgets.QTableWidgetItem()
            chk.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
            )
            chk.setCheckState(
                Qt.CheckState.Checked if c.bookmarked else Qt.CheckState.Unchecked
            )
            self._table.setItem(row, self._COL_BOOKMARK, chk)

            notes_item = QtWidgets.QTableWidgetItem(c.notes)
            self._table.setItem(row, self._COL_NOTES, notes_item)

        self._table.resizeColumnsToContents()

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
