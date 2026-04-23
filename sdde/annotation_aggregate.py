"""
Helpers for treating committed GT and paste annotations as one exportable set.
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from .attributes import compute_size_tag, default_attributes_dict
from .metadata_export import build_annotation_records


def combined_real_rows(
    gt_real_data: Sequence[Sequence[Any]],
    paste_real_data: Sequence[Sequence[Any]],
) -> list[list[Any]]:
    return [list(row) for row in gt_real_data] + [list(row) for row in paste_real_data]


def combined_box_attributes(
    gt_box_attributes: Sequence[Mapping[str, str]],
    paste_real_data: Sequence[Sequence[Any]],
) -> list[dict[str, str]]:
    gt_attrs = [dict(attrs) for attrs in gt_box_attributes]
    paste_attrs: list[dict[str, str]] = []
    for row in paste_real_data:
        attrs = default_attributes_dict()
        if len(row) >= 5:
            attrs["size_tag"] = compute_size_tag(
                float(row[1]),
                float(row[2]),
                float(row[3]),
                float(row[4]),
            )
        paste_attrs.append(attrs)
    return gt_attrs + paste_attrs


def combined_gt_boxes(
    gt_real_data: Sequence[Sequence[Any]],
    paste_real_data: Sequence[Sequence[Any]],
) -> list[tuple[str, float, float, float, float]]:
    rows = combined_real_rows(gt_real_data, paste_real_data)
    boxes: list[tuple[str, float, float, float, float]] = []
    for row in rows:
        if len(row) < 5:
            continue
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


def build_combined_annotation_records(
    *,
    image_path: str | None,
    image_width: int | None,
    image_height: int | None,
    gt_real_data: Sequence[Sequence[Any]],
    gt_box_attributes: Sequence[Mapping[str, str]],
    paste_real_data: Sequence[Sequence[Any]],
    object_list: Sequence[str],
    class_id_to_super: Mapping[int, str] | None = None,
) -> list[dict[str, Any]]:
    gt_records = build_annotation_records(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        real_data=gt_real_data,
        box_attributes=gt_box_attributes,
        object_list=object_list,
        class_id_to_super=class_id_to_super,
    )
    for rec in gt_records:
        rec["annotation_source"] = "gt"

    paste_records = build_annotation_records(
        image_path=image_path,
        image_width=image_width,
        image_height=image_height,
        real_data=paste_real_data,
        box_attributes=[{} for _ in paste_real_data],
        object_list=object_list,
        class_id_to_super=class_id_to_super,
    )
    for rec in paste_records:
        rec["annotation_source"] = "paste"

    return gt_records + paste_records
