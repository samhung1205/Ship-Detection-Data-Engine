"""Small tests for spec-driven main-window shortcut helpers."""

import sys
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.main_window import MyWidget  # noqa: E402


class _FakeCheckBox:
    def __init__(self, checked: bool = False) -> None:
        self._checked = checked

    def isChecked(self) -> bool:  # noqa: N802
        return self._checked

    def setChecked(self, value: bool) -> None:  # noqa: N802
        self._checked = value


class _FakeListController:
    def __init__(self) -> None:
        self.delete_calls = 0

    def delete_selected(self) -> None:
        self.delete_calls += 1


def test_toggle_hide_boxes_flips_checkbox_and_refreshes() -> None:
    calls: list[bool] = []
    dummy = type("Dummy", (), {})()
    dummy.hideBox = _FakeCheckBox(False)
    dummy.hideBbox = lambda cb: calls.append(cb.isChecked())

    MyWidget._toggle_hide_boxes(dummy)

    assert dummy.hideBox.isChecked() is True
    assert calls == [True]


def test_delete_selected_gt_box_delegates_to_list_controller() -> None:
    dummy = type("Dummy", (), {})()
    dummy._gt_list_controller = _FakeListController()

    MyWidget._delete_selected_gt_box(dummy)

    assert dummy._gt_list_controller.delete_calls == 1


def test_platform_shortcut_uses_meta_on_macos() -> None:
    with patch("gui.main_window.sys.platform", "darwin"):
        assert MyWidget._platform_shortcut("Ctrl+I") == "Ctrl+I"


def test_platform_shortcut_keeps_ctrl_on_non_macos() -> None:
    with patch("gui.main_window.sys.platform", "linux"):
        assert MyWidget._platform_shortcut("Ctrl+I") == "Ctrl+I"


def test_redo_shortcuts_match_platform_conventions() -> None:
    with patch("gui.main_window.sys.platform", "darwin"):
        assert MyWidget._redo_shortcuts() == ["Ctrl+Shift+Z"]
    with patch("gui.main_window.sys.platform", "win32"):
        assert MyWidget._redo_shortcuts() == ["Ctrl+Shift+Z", "Ctrl+Y"]
