"""
Dataset statistics dialog (PRD §17).

Shows annotation distribution summaries and allows export to JSON / CSV.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from PyQt6 import QtWidgets
from PyQt6.QtCore import Qt

from sdde.statistics import compute_dataset_stats, export_stats_csv, export_stats_json


class StatisticsDialog(QtWidgets.QDialog):
    """Modal dialog that displays computed dataset statistics."""

    def __init__(
        self,
        parent: QtWidgets.QWidget | None,
        *,
        records: list[dict[str, Any]],
        scope_label: str = "Current image",
        total_images_override: int | None = None,
        labeled_images_override: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dataset Statistics")
        self.resize(620, 520)
        self._scope_label = scope_label
        self._stats = compute_dataset_stats(
            records,
            total_images_override=total_images_override,
            labeled_images_override=labeled_images_override,
        )
        self._build_ui()

    @property
    def stats(self) -> dict[str, Any]:
        return self._stats

    def _build_ui(self) -> None:
        layout = QtWidgets.QVBoxLayout(self)

        s = self._stats
        scope_lbl = QtWidgets.QLabel(f"Scope: {self._scope_label}")
        scope_lbl.setStyleSheet("color: #666; margin-bottom: 2px;")
        layout.addWidget(scope_lbl)
        header = (
            f"Images: {s['total_images']}  |  "
            f"Labeled: {s.get('labeled_images', 0)}  |  "
            f"Annotations: {s['total_annotations']}  |  "
            f"Avg/Image: {s.get('avg_annotations_per_image', 0):.2f}"
        )
        lbl = QtWidgets.QLabel(header)
        lbl.setStyleSheet("font-weight: bold; font-size: 14px; margin-bottom: 6px;")
        layout.addWidget(lbl)

        tabs = QtWidgets.QTabWidget()

        tabs.addTab(self._make_overview_table(s), "Overview")
        tabs.addTab(self._make_count_table("Class", s.get("class_distribution", {})), "Class dist")
        tabs.addTab(self._make_count_table("Size tag", s.get("size_tag_distribution", {})), "Size dist")
        tabs.addTab(
            self._make_ratio_table(
                s.get("scene_tag_distribution", {}),
                s.get("scene_tag_ratio", {}),
            ),
            "Scene dist",
        )
        tabs.addTab(self._make_numeric_table(s), "Bbox stats")
        tabs.addTab(self._make_cross_table(s.get("class_x_size_distribution", {})), "Class × Size")

        layout.addWidget(tabs)

        btn_row = QtWidgets.QHBoxLayout()
        btn_json = QtWidgets.QPushButton("Export JSON…")
        btn_json.clicked.connect(self._export_json)
        btn_row.addWidget(btn_json)
        btn_csv = QtWidgets.QPushButton("Export CSV…")
        btn_csv.clicked.connect(self._export_csv)
        btn_row.addWidget(btn_csv)
        btn_row.addStretch()
        btn_close = QtWidgets.QPushButton("Close")
        btn_close.clicked.connect(self.accept)
        btn_row.addWidget(btn_close)
        layout.addLayout(btn_row)

    @staticmethod
    def _make_count_table(key_label: str, data: Mapping[str, int]) -> QtWidgets.QTableWidget:
        t = QtWidgets.QTableWidget()
        t.setColumnCount(2)
        t.setHorizontalHeaderLabels([key_label, "Count"])
        t.setRowCount(len(data))
        t.horizontalHeader().setStretchLastSection(True)
        for i, (k, v) in enumerate(data.items()):
            t.setItem(i, 0, _ro(k))
            t.setItem(i, 1, _ro(str(v)))
        t.resizeColumnsToContents()
        return t

    @staticmethod
    def _make_overview_table(stats: Mapping[str, Any]) -> QtWidgets.QTableWidget:
        rows = [
            ("Total images", stats.get("total_images", 0)),
            ("Labeled images", stats.get("labeled_images", 0)),
            ("Unlabeled images", stats.get("unlabeled_images", 0)),
            ("Total annotations", stats.get("total_annotations", 0)),
            ("Avg annotations / image", stats.get("avg_annotations_per_image", 0)),
            ("Distinct classes", len(stats.get("class_distribution", {}))),
        ]
        t = QtWidgets.QTableWidget()
        t.setColumnCount(2)
        t.setHorizontalHeaderLabels(["Metric", "Value"])
        t.setRowCount(len(rows))
        t.horizontalHeader().setStretchLastSection(True)
        for i, (label, value) in enumerate(rows):
            t.setItem(i, 0, _ro(str(label)))
            t.setItem(i, 1, _ro(str(value)))
        t.resizeColumnsToContents()
        return t

    @staticmethod
    def _make_ratio_table(
        counts: Mapping[str, int],
        ratios: Mapping[str, float],
    ) -> QtWidgets.QTableWidget:
        keys = list(counts.keys())
        t = QtWidgets.QTableWidget()
        t.setColumnCount(3)
        t.setHorizontalHeaderLabels(["Scene", "Count", "Ratio (%)"])
        t.setRowCount(len(keys))
        t.horizontalHeader().setStretchLastSection(True)
        for i, key in enumerate(keys):
            t.setItem(i, 0, _ro(str(key)))
            t.setItem(i, 1, _ro(str(counts.get(key, 0))))
            t.setItem(i, 2, _ro(str(ratios.get(key, 0))))
        t.resizeColumnsToContents()
        return t

    @staticmethod
    def _make_numeric_table(stats: Mapping[str, Any]) -> QtWidgets.QTableWidget:
        metrics = [
            ("Bbox width", "bbox_width_distribution"),
            ("Bbox height", "bbox_height_distribution"),
            ("Bbox area", "bbox_area_distribution"),
            ("Aspect ratio", "aspect_ratio_distribution"),
            ("Rotation angle", "rotation_angle_distribution"),
        ]
        t = QtWidgets.QTableWidget()
        cols = ["Metric", "Count", "Min", "Max", "Mean", "Std"]
        t.setColumnCount(len(cols))
        t.setHorizontalHeaderLabels(cols)
        t.setRowCount(len(metrics))
        t.horizontalHeader().setStretchLastSection(True)
        for i, (label, key) in enumerate(metrics):
            d = stats.get(key, {})
            t.setItem(i, 0, _ro(label))
            t.setItem(i, 1, _ro(str(d.get("count", 0))))
            t.setItem(i, 2, _ro(str(d.get("min", ""))))
            t.setItem(i, 3, _ro(str(d.get("max", ""))))
            t.setItem(i, 4, _ro(str(d.get("mean", ""))))
            t.setItem(i, 5, _ro(str(d.get("std", ""))))
        t.resizeColumnsToContents()
        return t

    @staticmethod
    def _make_cross_table(data: Mapping[str, Mapping[str, int]]) -> QtWidgets.QTableWidget:
        all_sizes = sorted({st for sizes in data.values() for st in sizes})
        classes = sorted(data.keys())
        t = QtWidgets.QTableWidget()
        t.setColumnCount(len(all_sizes) + 1)
        t.setHorizontalHeaderLabels(["Class"] + all_sizes)
        t.setRowCount(len(classes))
        t.horizontalHeader().setStretchLastSection(True)
        for i, cls in enumerate(classes):
            t.setItem(i, 0, _ro(cls))
            for j, st in enumerate(all_sizes):
                t.setItem(i, j + 1, _ro(str(data[cls].get(st, 0))))
        t.resizeColumnsToContents()
        return t

    def _export_json(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save statistics JSON", "stats.json", "JSON (*.json)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_stats_json(self._stats), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))

    def _export_csv(self) -> None:
        fp, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save statistics CSV", "stats.csv", "CSV (*.csv)"
        )
        if not fp:
            return
        try:
            Path(fp).write_text(export_stats_csv(self._stats), encoding="utf-8")
        except OSError as e:
            QtWidgets.QMessageBox.critical(self, "Export failed", str(e))


def _ro(text: str) -> QtWidgets.QTableWidgetItem:
    item = QtWidgets.QTableWidgetItem(text)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item
