"""
Tests for the GT annotation actions controller.
"""
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gui.annotation_actions_controller import AnnotationActionsController  # noqa: E402


class _FakeCanvas:
    def __init__(self, width: int, height: int) -> None:
        self._width = width
        self._height = height

    def width(self) -> int:
        return self._width

    def height(self) -> int:
        return self._height


class _FakeListView:
    def __init__(self) -> None:
        self._count = 0
        self.current_row = -1

    def count(self) -> int:
        return self._count

    def set_current_row(self, row: int) -> None:
        self.current_row = row


class _FakeCommandController:
    def __init__(self, list_view: _FakeListView) -> None:
        self.list_view = list_view
        self.add_calls = []
        self.remove_calls = []
        self.rename_calls = []
        self.update_calls = []
        self.append_calls = []
        self.clear_calls = 0

    def add_box(self, data_row, real_row, label_text, *, extended_object_list: bool) -> None:
        self.add_calls.append((data_row, real_row, label_text, extended_object_list))
        self.list_view._count += 1

    def remove_box(self, row: int) -> None:
        self.remove_calls.append(row)

    def rename_box(self, row: int, old_name: str, new_name: str) -> None:
        self.rename_calls.append((row, old_name, new_name))

    def update_box_geometry(self, row: int, data_row, real_row) -> None:
        self.update_calls.append((row, data_row, real_row))

    def clear_all_boxes(self) -> None:
        self.clear_calls += 1

    def append_blocks(self, blocks) -> None:
        self.append_calls.append(blocks)


def _make_controller(
    *,
    object_names=None,
    ask_label=None,
    on_add_cancelled=None,
):
    list_view = _FakeListView()
    command_controller = _FakeCommandController(list_view)
    controller = AnnotationActionsController(
        parent=None,
        command_controller=command_controller,
        list_view=list_view,
        get_object_names=lambda: list(object_names or []),
        get_canvas=lambda: _FakeCanvas(200, 100),
        get_origin_size=lambda: (1000, 500),
        ask_label=ask_label,
        on_add_cancelled=on_add_cancelled,
    )
    return controller, command_controller, list_view


def test_actions_controller_prompts_and_adds_box_from_rect() -> None:
    controller, command_controller, list_view = _make_controller(
        object_names=["naval"],
        ask_label=lambda _parent, _names: ("merchant", True),
    )

    added = controller.prompt_add_box(20, 10, 100, 50)

    assert added is True
    assert command_controller.add_calls == [
        (
            ["merchant", 20, 10, 100, 50, 200, 100],
            ["merchant", 100, 50, 500, 250],
            "merchant",
            True,
        )
    ]
    assert list_view.current_row == 0


def test_actions_controller_restores_view_when_add_prompt_is_cancelled() -> None:
    cancelled: list[str] = []
    controller, command_controller, list_view = _make_controller(
        ask_label=lambda _parent, _names: ("", False),
        on_add_cancelled=lambda: cancelled.append("cancelled"),
    )

    added = controller.prompt_add_box(20, 10, 100, 50)

    assert added is False
    assert cancelled == ["cancelled"]
    assert command_controller.add_calls == []
    assert list_view.current_row == -1


def test_actions_controller_builds_add_box_payload_from_prediction() -> None:
    controller, _command_controller, _list_view = _make_controller(object_names=["naval"])
    pred = SimpleNamespace(
        class_name="naval",
        x1=100,
        y1=50,
        x2=400,
        y2=250,
    )

    payload = controller.build_add_box_from_prediction(pred)

    assert payload == (
        ["naval", 20.0, 10.0, 80.0, 50.0, 200, 100],
        ["naval", 100, 50, 400, 250],
        "naval",
        False,
    )


def test_actions_controller_updates_existing_box_from_rect() -> None:
    controller, command_controller, list_view = _make_controller(object_names=["naval"])
    list_view._count = 1

    updated = controller.update_box_from_rect(
        row=0,
        item="naval",
        x1=20,
        y1=10,
        x2=100,
        y2=50,
        select_row=True,
    )

    assert updated is True
    assert command_controller.update_calls == [
        (
            0,
            ["naval", 20, 10, 100, 50, 200, 100],
            ["naval", 100, 50, 500, 250],
        )
    ]
    assert list_view.current_row == 0


def test_actions_controller_forwards_remove_rename_clear_and_append() -> None:
    controller, command_controller, _list_view = _make_controller()
    blocks = [(["naval", 1, 2, 3, 4, 200, 100], ["naval", 10, 20, 30, 40], "naval")]

    controller.remove_row(2)
    controller.rename_row(2, "naval", "merchant")
    controller.clear_all()
    controller.append_blocks(blocks)

    assert command_controller.remove_calls == [2]
    assert command_controller.rename_calls == [(2, "naval", "merchant")]
    assert command_controller.clear_calls == 1
    assert command_controller.append_calls == [blocks]
