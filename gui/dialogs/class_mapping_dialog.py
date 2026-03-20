"""
Edit class_id / class_name / super_category and save classes.yaml (PRD §6).
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, List

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QDialogButtonBox, QHeaderView, QMessageBox

from sdde.class_catalog import ClassCatalog, ClassInfo
from sdde.classes_yaml import save_classes_yaml_path

from ..class_mapping_service import default_classes_yaml_path

if TYPE_CHECKING:
    from ..main_window import MyWidget


class ClassMappingDialog(QtWidgets.QDialog):
    def __init__(self, main_widget: "MyWidget", parent=None):
        super().__init__(parent)
        self.main_widget = main_widget
        self.setWindowTitle("Class mapping (SDDE)")
        self.resize(520, 360)
        self._orig_sig = main_widget.class_catalog.signature()
        lay = QtWidgets.QVBoxLayout(self)

        row = QtWidgets.QHBoxLayout()
        row.addWidget(QtWidgets.QLabel("Project name:"))
        self.project_name = QtWidgets.QLineEdit(main_widget.class_catalog.project_name)
        row.addWidget(self.project_name)
        lay.addLayout(row)

        self.table = QtWidgets.QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(
            ["class_id", "class_name", "super_category"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        lay.addWidget(self.table)

        btn_row = QtWidgets.QHBoxLayout()
        self.btn_add = QtWidgets.QPushButton("Add class")
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove = QtWidgets.QPushButton("Remove selected")
        self.btn_remove.clicked.connect(self._remove_selected_row)
        self.btn_load = QtWidgets.QPushButton("Load YAML…")
        self.btn_load.clicked.connect(self._load_yaml)
        self.btn_save = QtWidgets.QPushButton("Save YAML…")
        self.btn_save.clicked.connect(self._save_yaml_as)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_save)
        lay.addLayout(btn_row)

        box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        box.accepted.connect(self._on_ok)
        box.rejected.connect(self.reject)
        lay.addWidget(box)

        self._fill_table_from_catalog(main_widget.class_catalog)

    def _fill_table_from_catalog(self, cat: ClassCatalog) -> None:
        self.project_name.setText(cat.project_name)
        self.table.setRowCount(0)
        for c in sorted(cat.classes, key=lambda x: x.class_id):
            self._append_row(c.class_id, c.class_name, c.super_category)

    def _append_row(self, class_id: int, name: str, super_cat: str) -> None:
        r = self.table.rowCount()
        self.table.insertRow(r)
        self.table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(class_id)))
        self.table.setItem(r, 1, QtWidgets.QTableWidgetItem(name))
        self.table.setItem(r, 2, QtWidgets.QTableWidgetItem(super_cat))

    def _add_row(self) -> None:
        ids = self._read_ids_from_table()
        next_id = max(ids, default=-1) + 1
        self._append_row(next_id, "new_class", "vessel")

    def _remove_selected_row(self) -> None:
        r = self.table.currentRow()
        if r < 0:
            return
        self.table.removeRow(r)

    def _read_ids_from_table(self) -> List[int]:
        ids: List[int] = []
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is None:
                continue
            ids.append(int(it.text().strip()))
        return ids

    def _catalog_from_table(self) -> ClassCatalog:
        rows: list[ClassInfo] = []
        for r in range(self.table.rowCount()):
            id_item = self.table.item(r, 0)
            n_item = self.table.item(r, 1)
            s_item = self.table.item(r, 2)
            if id_item is None or n_item is None or s_item is None:
                continue
            rows.append(
                ClassInfo(
                    class_id=int(id_item.text().strip()),
                    class_name=n_item.text().strip(),
                    super_category=s_item.text().strip(),
                )
            )
        return ClassCatalog.from_list(self.project_name.text().strip() or "project", rows)

    def _load_yaml(self) -> None:
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Load classes.yaml", str(Path.home()), "YAML (*.yaml *.yml)"
        )
        if path:
            try:
                from sdde.classes_yaml import load_classes_yaml_path

                cat = load_classes_yaml_path(path)
                self._fill_table_from_catalog(cat)
            except Exception as e:
                QMessageBox.critical(self, "Load failed", str(e))

    def _save_yaml_as(self) -> None:
        try:
            cat = self._catalog_from_table()
            cat.validate()
        except Exception as e:
            QMessageBox.warning(self, "Invalid mapping", str(e))
            return
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, "Save classes.yaml", str(default_classes_yaml_path()), "YAML (*.yaml)"
        )
        if path:
            try:
                save_classes_yaml_path(cat, path)
            except OSError as e:
                QMessageBox.critical(self, self.windowTitle(), str(e))

    def _on_ok(self) -> None:
        try:
            new_cat = self._catalog_from_table()
            new_cat.validate()
        except Exception as e:
            QMessageBox.warning(self, "Invalid mapping", str(e))
            return

        if new_cat.signature() != self._orig_sig and self._has_box_annotations():
            r = QMessageBox.warning(
                self,
                "Mapping change",
                "Existing box annotations use the previous class names / ids.\n"
                "Changing the mapping may make labels inconsistent with YOLO files.\n\n"
                "Continue?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if r != QMessageBox.StandardButton.Ok:
                return

        self.main_widget.apply_class_catalog(new_cat)
        try:
            save_classes_yaml_path(new_cat, default_classes_yaml_path())
        except OSError:
            pass
        self.accept()

    def _has_box_annotations(self) -> bool:
        return bool(self.main_widget.data or self.main_widget.real_data)
