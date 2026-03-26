"""
Transitional session container for in-progress paste candidate state.

This centralises the temporary state that exists between:

- choosing an RGBA asset,
- adjusting resize / rotate / brightness / contrast / flip,
- clicking a placement anchor on the canvas,
- and committing the candidate into ``PasteDocument``.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class PasteCandidateSession:
    asset_path: str = ""

    pasteimg: Any = None
    origin_pasteimg: Any = None
    paste_canvas: Any = None
    pasteimg_width: int = 0
    pasteimg_height: int = 0
    pasteimg_channel: int = 0

    resizeimg: Any = None
    rotated: Any = None
    bc_image: Any = None
    norm_pimg: list[Any] | None = None
    bbox_pimg: list[Any] | None = None
    real_bbox_pimg: list[Any] | None = None
    pasteimg_canvas: Any = None

    anchor_x: int | None = None
    anchor_y: int | None = None

    @property
    def has_anchor(self) -> bool:
        return self.anchor_x is not None and self.anchor_y is not None

    def set_anchor(self, x: int, y: int) -> None:
        self.anchor_x = x
        self.anchor_y = y

    def clear_candidate(self) -> None:
        self.resizeimg = None
        self.rotated = None
        self.bc_image = None
        self.norm_pimg = None
        self.bbox_pimg = None
        self.real_bbox_pimg = None
        self.pasteimg_canvas = None
        self.anchor_x = None
        self.anchor_y = None

    def clear(self) -> None:
        self.asset_path = ""
        self.pasteimg = None
        self.origin_pasteimg = None
        self.paste_canvas = None
        self.pasteimg_width = 0
        self.pasteimg_height = 0
        self.pasteimg_channel = 0
        self.clear_candidate()
