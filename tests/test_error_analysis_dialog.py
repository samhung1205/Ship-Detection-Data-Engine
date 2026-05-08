"""Tests for error analysis dialog rendering behavior."""

import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6 import QtWidgets  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.dialogs.error_analysis_dialog import ErrorAnalysisDialog  # noqa: E402
from sdde.error_analysis import ErrorCase  # noqa: E402


APP = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


def test_error_analysis_dialog_limits_large_table_render(monkeypatch) -> None:
    monkeypatch.setattr(ErrorAnalysisDialog, "MAX_TABLE_ROWS", 3)
    cases = [
        ErrorCase(image_id=f"/tmp/img_{idx}.jpg", error_type="FP", pred_class="naval")
        for idx in range(5)
    ]

    dialog = ErrorAnalysisDialog(
        None,
        gt_boxes=[],
        predictions=[],
        cases=cases,
    )
    APP.processEvents()

    assert dialog._table.rowCount() == 3
    assert "Table limited to first 3" in dialog._summary_label.text()
    assert len(dialog.cases) == 5
