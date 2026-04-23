"""
Tests for the GT workspace controller.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.document import AnnotationDocument  # noqa: E402
from gui.annotation_workspace_controller import AnnotationWorkspaceController  # noqa: E402


class _Combo:
    def __init__(self) -> None:
        self.text = ""

    def setCurrentText(self, text: str) -> None:  # noqa: N802
        self.text = text


class _AttrPanel:
    def __init__(self) -> None:
        self.enabled = False
        self.loaded = None
        self.values = {
            "size_tag": "medium",
            "crowded": "false",
            "difficulty_tag": "normal",
            "hard_sample": "false",
            "occluded": "false",
            "truncated": "false",
            "blurred": "false",
            "difficult_background": "false",
            "low_contrast": "false",
            "scene_tag": "unknown",
        }
        self.combo_size = _Combo()

    def set_enabled_editing(self, enabled: bool) -> None:
        self.enabled = enabled

    def load_from_dict(self, attrs: dict[str, str]) -> None:
        self.loaded = dict(attrs)
        self.values = dict(attrs)

    def to_dict(self) -> dict[str, str]:
        out = dict(self.values)
        if self.combo_size.text:
            out["size_tag"] = self.combo_size.text
        return out


class _ListView:
    def __init__(self) -> None:
        self.row = -1

    def current_row(self) -> int:
        return self.row

    def current_index_row(self) -> int:
        return self.row

    def set_current_row(self, row: int) -> None:
        self.row = row


class _PreviewController:
    def __init__(self) -> None:
        self.preview_calls: list[int] = []
        self.clear_count = 0

    def preview_row(self, row: int) -> None:
        self.preview_calls.append(row)

    def clear_preview(self) -> None:
        self.clear_count += 1


def _make_controller():
    doc = AnnotationDocument()
    doc.append_box(
        ["naval", 1, 2, 3, 4, 100, 100],
        ["naval", 10, 20, 30, 40],
    )
    list_view = _ListView()
    panel = _AttrPanel()
    preview = _PreviewController()
    delete_calls: list[int] = []
    rename_calls: list[tuple[int, str, str]] = []
    clear_calls: list[str] = []
    controller = AnnotationWorkspaceController(
        document=doc,
        list_view=list_view,
        attr_panel=panel,
        preview_controller=preview,
        on_delete_row=delete_calls.append,
        on_rename_row=lambda row, old, new: rename_calls.append((row, old, new)),
        on_clear_all=lambda: clear_calls.append("clear"),
    )
    return controller, doc, list_view, panel, preview, delete_calls, rename_calls, clear_calls


def test_workspace_controller_loads_attrs_for_selected_row() -> None:
    controller, _doc, list_view, panel, preview, *_rest = _make_controller()

    list_view.set_current_row(0)
    controller.on_row_changed(0)

    assert panel.enabled is True
    assert panel.loaded is not None
    assert panel.loaded["size_tag"] == "small"
    assert preview.preview_calls == [0]


def test_workspace_controller_updates_document_from_attr_panel() -> None:
    controller, doc, list_view, panel, *_rest = _make_controller()

    list_view.set_current_row(0)
    panel.values["crowded"] = "true"
    panel.values["hard_sample"] = "true"
    panel.values["occluded"] = "true"
    controller.on_attr_panel_changed()

    assert doc.box_attributes[0]["crowded"] == "true"
    assert doc.box_attributes[0]["hard_sample"] == "true"
    assert doc.box_attributes[0]["occluded"] == "true"


def test_workspace_controller_recalculates_size_tag_for_selection() -> None:
    controller, doc, list_view, panel, *_rest = _make_controller()

    list_view.set_current_row(0)
    controller.on_recalc_size_tag()

    assert panel.combo_size.text == "small"
    assert doc.box_attributes[0]["size_tag"] == "small"


def test_workspace_controller_previews_and_clears_selection() -> None:
    controller, _doc, list_view, panel, preview, *_rest = _make_controller()

    list_view.set_current_row(0)
    controller.preview_current_row()
    controller.clear_selection()

    assert preview.preview_calls == [0]
    assert preview.clear_count == 1
    assert list_view.current_row() == -1
    assert panel.enabled is False


def test_workspace_controller_forwards_delete_rename_and_clear() -> None:
    controller, _doc, _list_view, _panel, preview, delete_calls, rename_calls, clear_calls = _make_controller()

    controller.delete_row(0)
    controller.rename_row(0, "naval", "merchant")
    controller.clear_all()

    assert preview.clear_count == 2
    assert delete_calls == [0]
    assert rename_calls == [(0, "naval", "merchant")]
    assert clear_calls == ["clear"]


def test_workspace_controller_clears_preview_for_invalid_selection() -> None:
    controller, _doc, _list_view, panel, preview, *_rest = _make_controller()

    controller.on_row_changed(-1)

    assert preview.clear_count == 1
    assert panel.enabled is False
