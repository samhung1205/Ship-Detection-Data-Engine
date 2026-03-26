"""
Transitional document container for committed paste-image state.

This centralises the legacy parallel paste lists used by the GUI:

- ``pimg_data``: canvas/display-space paste bbox rows
- ``real_pimg_data``: image-pixel paste bbox rows
- ``paste_images``: committed RGBA overlay payloads
- ``paste_records``: augmentation metadata records aligned by index when present
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Sequence


def _clone_paste_image(paste_image: Any) -> Any:
    """
    Clone one committed paste overlay payload without deepcopying Qt objects.

    The legacy payload shape is typically ``[QImage, norm_x, norm_y, norm_w, norm_h]``.
    We only need container isolation here; the inner image object may be a Qt type
    such as ``QImage`` that cannot be pickled by ``copy.deepcopy``.
    """
    if isinstance(paste_image, (list, tuple)):
        return list(paste_image)
    return paste_image


@dataclass(frozen=True)
class PasteEntryState:
    """Cloned snapshot for one committed paste entry."""

    data_row: list[Any]
    real_row: list[Any]
    paste_image: Any
    paste_record: Any | None = None


@dataclass
class PasteDocument:
    """
    Transitional owner for one image's committed paste state.

    The row shapes intentionally stay compatible with the legacy GUI so the
    preview and transform code can be migrated in smaller steps later.
    """

    pimg_data: list[list[Any]] = field(default_factory=list)
    real_pimg_data: list[list[Any]] = field(default_factory=list)
    paste_images: list[Any] = field(default_factory=list)
    paste_records: list[Any] = field(default_factory=list)

    @property
    def total_pastes(self) -> int:
        return len(self.real_pimg_data)

    def clear(self) -> None:
        """Clear committed paste rows in place so existing references stay valid."""
        self.pimg_data.clear()
        self.real_pimg_data.clear()
        self.paste_images.clear()
        self.paste_records.clear()

    def replace(
        self,
        *,
        pimg_data: Sequence[Sequence[Any]] | None = None,
        real_pimg_data: Sequence[Sequence[Any]] | None = None,
        paste_images: Sequence[Any] | None = None,
        paste_records: Sequence[Any] | None = None,
    ) -> None:
        """Replace one or more paste sections while preserving list identity."""
        if pimg_data is not None:
            self.pimg_data[:] = [list(row) for row in pimg_data]
        if real_pimg_data is not None:
            self.real_pimg_data[:] = [list(row) for row in real_pimg_data]
        if paste_images is not None:
            self.paste_images[:] = [_clone_paste_image(image) for image in paste_images]
        if paste_records is not None:
            self.paste_records[:] = list(paste_records)

    def validate_alignment(self) -> None:
        """Raise if the committed paste lists are no longer index-aligned."""
        if len(self.pimg_data) != len(self.real_pimg_data):
            raise ValueError(
                "PasteDocument alignment error: "
                f"len(pimg_data)={len(self.pimg_data)} "
                f"!= len(real_pimg_data)={len(self.real_pimg_data)}"
            )
        if len(self.paste_images) != len(self.real_pimg_data):
            raise ValueError(
                "PasteDocument alignment error: "
                f"len(paste_images)={len(self.paste_images)} "
                f"!= len(real_pimg_data)={len(self.real_pimg_data)}"
            )
        if len(self.paste_records) > len(self.real_pimg_data):
            raise ValueError(
                "PasteDocument alignment error: "
                f"len(paste_records)={len(self.paste_records)} "
                f"> len(real_pimg_data)={len(self.real_pimg_data)}"
            )

    def append_paste(
        self,
        data_row: Sequence[Any],
        real_row: Sequence[Any],
        paste_image: Any,
        *,
        paste_record: Any | None = None,
    ) -> PasteEntryState:
        state = PasteEntryState(
            data_row=list(data_row),
            real_row=list(real_row),
            paste_image=_clone_paste_image(paste_image),
            paste_record=copy.deepcopy(paste_record),
        )
        self.pimg_data.append(copy.deepcopy(state.data_row))
        self.real_pimg_data.append(copy.deepcopy(state.real_row))
        self.paste_images.append(_clone_paste_image(state.paste_image))
        if paste_record is not None:
            self.paste_records.append(copy.deepcopy(state.paste_record))
        self.validate_alignment()
        return state

    def remove_paste(self, index: int) -> PasteEntryState:
        state = PasteEntryState(
            data_row=copy.deepcopy(self.pimg_data.pop(index)),
            real_row=copy.deepcopy(self.real_pimg_data.pop(index)),
            paste_image=_clone_paste_image(self.paste_images.pop(index)),
            paste_record=copy.deepcopy(
                self.paste_records.pop(index) if index < len(self.paste_records) else None
            ),
        )
        self.validate_alignment()
        return state

    def rename_paste(self, index: int, new_name: str) -> str:
        old_name = str(self.real_pimg_data[index][0])
        self.pimg_data[index][0] = new_name
        self.real_pimg_data[index][0] = new_name
        if index < len(self.paste_records):
            self.paste_records[index].class_name = new_name
        return old_name
