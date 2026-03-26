"""
Adapters between legacy GUI row shapes and SDDE annotation models.

These helpers let the PyQt widgets reuse the service-layer import/export logic
without changing the legacy in-memory structures all at once.
"""
from __future__ import annotations

from typing import Any, Sequence

from .models import ClassMapping, HBBAnnotation, HBBBoxPx


def class_mapping_from_object_list(object_list: Sequence[str]) -> ClassMapping:
    """Build a validated ClassMapping from the GUI object_list order."""
    mapping = ClassMapping(names=[str(name) for name in object_list])
    mapping.validate()
    return mapping


def annotations_from_legacy_rows(
    rows: Sequence[Sequence[Any]],
    *,
    object_list: Sequence[str],
) -> list[HBBAnnotation]:
    """
    Convert legacy rows ``[name, x1, y1, x2, y2]`` into HBBAnnotation objects.
    """
    mapping = class_mapping_from_object_list(object_list)
    annotations: list[HBBAnnotation] = []
    for row in rows:
        if len(row) < 5:
            raise ValueError("Legacy annotation row must contain at least 5 values")
        name = str(row[0])
        cls_id = mapping.name_to_id(name)
        annotations.append(
            HBBAnnotation(
                class_id=cls_id,
                bbox_px=HBBBoxPx(
                    x1=float(row[1]),
                    y1=float(row[2]),
                    x2=float(row[3]),
                    y2=float(row[4]),
                ),
            )
        )
    return annotations


def legacy_blocks_from_annotations(
    annotations: Sequence[HBBAnnotation],
    *,
    class_mapping: ClassMapping,
    canvas_w: int,
    canvas_h: int,
) -> list[tuple[list[Any], list[Any], str]]:
    """
    Convert annotations into ``BulkAppendBoxesCommand`` blocks.

    Each block is ``(data_row, real_row, list_label)`` using the legacy row
    shapes expected by ``gui.main_window``.
    """
    class_mapping.validate()
    if canvas_w <= 0 or canvas_h <= 0:
        raise ValueError("canvas_w/canvas_h must be positive")

    blocks: list[tuple[list[Any], list[Any], str]] = []
    for ann in annotations:
        name = class_mapping.id_to_name(ann.class_id)
        x1 = int(ann.bbox_px.x1)
        y1 = int(ann.bbox_px.y1)
        x2 = int(ann.bbox_px.x2)
        y2 = int(ann.bbox_px.y2)
        data_row = [name, x1, y1, x2, y2, canvas_w, canvas_h]
        real_row = [name, x1, y1, x2, y2]
        blocks.append((data_row, real_row, name))
    return blocks
