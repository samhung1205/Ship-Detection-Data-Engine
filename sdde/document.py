"""
Transitional GT document container for the main-window refactor.

This module centralises the legacy parallel GT lists used by the GUI:

- ``data``: canvas/display-space bbox rows
- ``real_data``: image-pixel bbox rows
- ``box_attributes``: per-box attribute dicts aligned by index

The goal of this first step is intentionally modest: give the GUI a single
owner for annotation truth state without forcing a big UI rewrite.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class AnnotationBoxState:
    """Deep-copied snapshot for one GT box row trio."""

    data_row: list[Any]
    real_row: list[Any]
    attributes: dict[str, str]


@dataclass(frozen=True)
class AnnotationDocumentSnapshot:
    """Deep-copied snapshot for the whole GT document."""

    data: list[list[Any]]
    real_data: list[list[Any]]
    box_attributes: list[dict[str, str]]


@dataclass
class AnnotationDocument:
    """
    Transitional owner for one image's GT annotation state.

    The row shapes intentionally stay compatible with the legacy GUI so that
    existing widgets, dialogs, and command objects can be migrated in small
    increments.
    """

    data: list[list[Any]] = field(default_factory=list)
    real_data: list[list[Any]] = field(default_factory=list)
    box_attributes: list[dict[str, str]] = field(default_factory=list)

    @property
    def total_boxes(self) -> int:
        return len(self.real_data)

    @staticmethod
    def _normalise_attributes(
        real_row: Sequence[Any],
        attributes: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        from sdde.attributes import compute_size_tag, default_attributes_dict

        x1, y1, x2, y2 = (
            float(real_row[1]),
            float(real_row[2]),
            float(real_row[3]),
            float(real_row[4]),
        )
        defaults = default_attributes_dict()
        defaults["size_tag"] = compute_size_tag(x1, y1, x2, y2)
        if attributes is not None:
            defaults.update(dict(attributes))
        return defaults

    def clear(self) -> None:
        """Clear GT rows in place so existing references stay valid."""
        self.data.clear()
        self.real_data.clear()
        self.box_attributes.clear()

    def replace(
        self,
        *,
        data: Sequence[Sequence[Any]] | None = None,
        real_data: Sequence[Sequence[Any]] | None = None,
        box_attributes: Sequence[Mapping[str, str]] | None = None,
    ) -> None:
        """Replace one or more GT sections while preserving list identity."""
        if data is not None:
            self.data[:] = [list(row) for row in data]
        if real_data is not None:
            self.real_data[:] = [list(row) for row in real_data]
        if box_attributes is not None:
            self.box_attributes[:] = [dict(attr) for attr in box_attributes]

    def validate_alignment(self) -> None:
        """Raise if the legacy parallel GT lists are no longer index-aligned."""
        if len(self.data) != len(self.real_data):
            raise ValueError(
                "AnnotationDocument alignment error: "
                f"len(data)={len(self.data)} != len(real_data)={len(self.real_data)}"
            )
        if len(self.box_attributes) != len(self.real_data):
            raise ValueError(
                "AnnotationDocument alignment error: "
                f"len(box_attributes)={len(self.box_attributes)} "
                f"!= len(real_data)={len(self.real_data)}"
            )

    def append_box(
        self,
        data_row: Sequence[Any],
        real_row: Sequence[Any],
        *,
        attributes: Mapping[str, str] | None = None,
    ) -> AnnotationBoxState:
        state = AnnotationBoxState(
            data_row=list(data_row),
            real_row=list(real_row),
            attributes=self._normalise_attributes(real_row, attributes),
        )
        self.data.append(copy.deepcopy(state.data_row))
        self.real_data.append(copy.deepcopy(state.real_row))
        self.box_attributes.append(copy.deepcopy(state.attributes))
        self.validate_alignment()
        return state

    def insert_box(
        self,
        index: int,
        state: AnnotationBoxState,
    ) -> None:
        self.data.insert(index, copy.deepcopy(state.data_row))
        self.real_data.insert(index, copy.deepcopy(state.real_row))
        self.box_attributes.insert(index, copy.deepcopy(state.attributes))
        self.validate_alignment()

    def remove_box(self, index: int) -> AnnotationBoxState:
        state = AnnotationBoxState(
            data_row=copy.deepcopy(self.data.pop(index)),
            real_row=copy.deepcopy(self.real_data.pop(index)),
            attributes=copy.deepcopy(self.box_attributes.pop(index)),
        )
        self.validate_alignment()
        return state

    def box_state(self, index: int) -> AnnotationBoxState:
        return AnnotationBoxState(
            data_row=copy.deepcopy(self.data[index]),
            real_row=copy.deepcopy(self.real_data[index]),
            attributes=copy.deepcopy(self.box_attributes[index]),
        )

    def replace_box(
        self,
        index: int,
        state: AnnotationBoxState,
    ) -> None:
        self.data[index] = copy.deepcopy(state.data_row)
        self.real_data[index] = copy.deepcopy(state.real_row)
        self.box_attributes[index] = copy.deepcopy(state.attributes)
        self.validate_alignment()

    def rename_box(self, index: int, new_name: str) -> str:
        old_name = str(self.real_data[index][0])
        self.data[index][0] = new_name
        self.real_data[index][0] = new_name
        return old_name

    def attributes_or_default(self, index: int) -> dict[str, str]:
        from sdde.attributes import default_attributes_dict

        if index < 0 or index >= len(self.box_attributes):
            return default_attributes_dict()
        return dict(self.box_attributes[index])

    def set_box_attributes(
        self,
        index: int,
        attributes: Mapping[str, str],
    ) -> None:
        self.box_attributes[index] = self._normalise_attributes(
            self.real_data[index],
            attributes,
        )
        self.validate_alignment()

    def apply_box_attributes(
        self,
        box_attributes: Sequence[Mapping[str, str]],
    ) -> None:
        self.box_attributes[:] = [
            self._normalise_attributes(
                row,
                box_attributes[i] if i < len(box_attributes) else None,
            )
            for i, row in enumerate(self.real_data)
        ]
        self.validate_alignment()

    def recalc_size_tag(self, index: int) -> str:
        from sdde.attributes import compute_size_tag

        attrs = self.attributes_or_default(index)
        row = self.real_data[index]
        attrs["size_tag"] = compute_size_tag(
            float(row[1]),
            float(row[2]),
            float(row[3]),
            float(row[4]),
        )
        self.box_attributes[index] = attrs
        return attrs["size_tag"]

    def gt_boxes(self) -> list[tuple[str, float, float, float, float]]:
        boxes: list[tuple[str, float, float, float, float]] = []
        for row in self.real_data:
            boxes.append(
                (
                    str(row[0]),
                    float(row[1]),
                    float(row[2]),
                    float(row[3]),
                    float(row[4]),
                )
            )
        return boxes

    def append_boxes(
        self,
        blocks: Sequence[tuple[Sequence[Any], Sequence[Any], str]],
    ) -> None:
        for data_row, real_row, _label in blocks:
            self.append_box(data_row, real_row)

    def snapshot(self) -> AnnotationDocumentSnapshot:
        return AnnotationDocumentSnapshot(
            data=copy.deepcopy(self.data),
            real_data=copy.deepcopy(self.real_data),
            box_attributes=copy.deepcopy(self.box_attributes),
        )

    def restore(self, snapshot: AnnotationDocumentSnapshot) -> None:
        self.replace(
            data=snapshot.data,
            real_data=snapshot.real_data,
            box_attributes=snapshot.box_attributes,
        )
        self.validate_alignment()

    def build_metadata_records(
        self,
        *,
        image_path: str | None,
        image_width: int | None,
        image_height: int | None,
        object_list: Sequence[str],
        class_id_to_super: Mapping[int, str] | None = None,
    ) -> list[dict[str, Any]]:
        from .metadata_export import build_annotation_records

        return build_annotation_records(
            image_path=image_path,
            image_width=image_width,
            image_height=image_height,
            real_data=self.real_data,
            box_attributes=self.box_attributes,
            object_list=object_list,
            class_id_to_super=class_id_to_super,
        )
