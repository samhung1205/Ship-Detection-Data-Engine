"""
Dataset/folder annotation scanning helpers for analysis workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2

from .import_export import import_json_label_file, import_yolo_hbb_label_file
from .image_browser import list_supported_images
from .legacy_rows import class_mapping_from_object_list
from .metadata_export import build_annotation_records


@dataclass(frozen=True)
class FolderAnnotationScanResult:
    folder_path: str
    image_paths: tuple[str, ...]
    labeled_image_paths: tuple[str, ...]
    records: tuple[dict[str, Any], ...]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def labeled_images(self) -> int:
        return len(self.labeled_image_paths)


@dataclass(frozen=True)
class ImageAnnotationBundle:
    image_path: str
    image_width: int
    image_height: int
    records: tuple[dict[str, Any], ...]
    gt_boxes: tuple[tuple[str, float, float, float, float], ...]
    gt_attributes: tuple[dict[str, str], ...]
    has_label: bool


def scan_folder_annotation_records(
    folder: str | Path,
    *,
    object_list: Sequence[str],
    class_id_to_super: Mapping[int, str] | None = None,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
    current_image_path: str | Path | None = None,
    current_image_records: Sequence[Mapping[str, Any]] | None = None,
) -> FolderAnnotationScanResult:
    folder_path = Path(folder)
    image_paths = tuple(list_supported_images(folder_path))
    current_image_resolved = _resolve_path_or_none(current_image_path)
    label_image_paths: list[str] = []
    records: list[dict[str, Any]] = []

    for image_path_str in image_paths:
        image_path = Path(image_path_str).resolve()
        if current_image_resolved is not None and image_path == current_image_resolved:
            if current_image_records:
                records.extend(dict(rec) for rec in current_image_records)
                label_image_paths.append(str(image_path))
            continue

        image_records, has_label = load_image_annotation_records(
            image_path,
            object_list=object_list,
            class_id_to_super=class_id_to_super,
            image_root=image_root,
            label_root=label_root,
        )
        if has_label:
            label_image_paths.append(str(image_path))
        records.extend(image_records)

    return FolderAnnotationScanResult(
        folder_path=str(folder_path),
        image_paths=image_paths,
        labeled_image_paths=tuple(label_image_paths),
        records=tuple(records),
    )


def load_image_annotation_records(
    image_path: str | Path,
    *,
    object_list: Sequence[str],
    class_id_to_super: Mapping[int, str] | None = None,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
) -> tuple[list[dict[str, Any]], bool]:
    bundle = load_image_annotation_bundle(
        image_path,
        object_list=object_list,
        class_id_to_super=class_id_to_super,
        image_root=image_root,
        label_root=label_root,
    )
    return list(bundle.records), bundle.has_label


def load_image_annotation_bundle(
    image_path: str | Path,
    *,
    object_list: Sequence[str],
    class_id_to_super: Mapping[int, str] | None = None,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
) -> ImageAnnotationBundle:
    image = Path(image_path)
    image_size = read_image_size(image)
    if image_size is None:
        return ImageAnnotationBundle(
            image_path=str(image),
            image_width=0,
            image_height=0,
            records=(),
            gt_boxes=(),
            gt_attributes=(),
            has_label=False,
        )
    image_w, image_h = image_size
    class_mapping = class_mapping_from_object_list(object_list)

    for label_path in _candidate_label_paths(
        image,
        image_root=image_root,
        label_root=label_root,
    ):
        if not label_path.is_file():
            continue
        try:
            if label_path.suffix.lower() == ".txt":
                annotations = import_yolo_hbb_label_file(
                    label_path,
                    class_mapping=class_mapping,
                    image_w=image_w,
                    image_h=image_h,
                )
            elif label_path.suffix.lower() == ".json":
                annotations = import_json_label_file(
                    label_path,
                    class_mapping=class_mapping,
                    image_w=image_w,
                    image_h=image_h,
                    image_path=image,
                )
            else:
                continue
        except (OSError, ValueError, TypeError, KeyError, IndexError):
            continue
        records, gt_boxes, gt_attributes = _build_annotation_payloads(
            image_path=image,
            image_width=image_w,
            image_height=image_h,
            object_list=object_list,
            class_id_to_super=class_id_to_super,
            annotations=annotations,
            class_mapping=class_mapping,
        )
        return ImageAnnotationBundle(
            image_path=str(image),
            image_width=image_w,
            image_height=image_h,
            records=tuple(records),
            gt_boxes=tuple(gt_boxes),
            gt_attributes=tuple(gt_attributes),
            has_label=True,
        )
    return ImageAnnotationBundle(
        image_path=str(image),
        image_width=image_w,
        image_height=image_h,
        records=(),
        gt_boxes=(),
        gt_attributes=(),
        has_label=False,
    )


def _candidate_label_paths(
    image_path: Path,
    *,
    image_root: str | Path | None,
    label_root: str | Path | None,
) -> list[Path]:
    bases: list[Path] = []
    mapped_base = _mapped_label_base(
        image_path,
        image_root=image_root,
        label_root=label_root,
    )
    if mapped_base is not None:
        bases.append(mapped_base)
    bases.append(image_path.with_suffix(""))

    candidates: list[Path] = []
    seen: set[str] = set()
    for base in bases:
        for suffix in (".json", ".txt"):
            candidate = base.with_suffix(suffix)
            key = str(candidate.resolve()) if candidate.exists() else str(candidate)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
    return candidates


def _mapped_label_base(
    image_path: Path,
    *,
    image_root: str | Path | None,
    label_root: str | Path | None,
) -> Path | None:
    if not label_root:
        return None
    label_root_path = Path(label_root)
    if image_root:
        try:
            rel = image_path.resolve().relative_to(Path(image_root).resolve())
        except ValueError:
            rel = None
        if rel is not None:
            return (label_root_path / rel.parent / rel.stem)
    return label_root_path / image_path.stem


def read_image_size(path: str | Path) -> tuple[int, int] | None:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None or len(img.shape) < 2:
        return None
    height, width = int(img.shape[0]), int(img.shape[1])
    return width, height


def _build_annotation_payloads(
    *,
    image_path: str | Path,
    image_width: int,
    image_height: int,
    object_list: Sequence[str],
    class_id_to_super: Mapping[int, str] | None,
    annotations: Sequence[Any],
    class_mapping=None,
) -> tuple[list[dict[str, Any]], list[tuple[str, float, float, float, float]], list[dict[str, str]]]:
    class_mapping = class_mapping or class_mapping_from_object_list(object_list)
    real_rows: list[list[Any]] = []
    attrs: list[dict[str, str]] = []
    gt_boxes: list[tuple[str, float, float, float, float]] = []
    for ann in annotations:
        name = class_mapping.id_to_name(int(ann.class_id))
        x1 = float(ann.bbox_px.x1)
        y1 = float(ann.bbox_px.y1)
        x2 = float(ann.bbox_px.x2)
        y2 = float(ann.bbox_px.y2)
        real_rows.append(
            [
                name,
                x1,
                y1,
                x2,
                y2,
            ]
        )
        gt_boxes.append((name, x1, y1, x2, y2))
        attrs.append(dict(getattr(ann, "attributes", {}) or {}))
    records = build_annotation_records(
        image_path=str(image_path),
        image_width=image_width,
        image_height=image_height,
        real_data=real_rows,
        box_attributes=attrs,
        object_list=object_list,
        class_id_to_super=class_id_to_super,
    )
    return records, gt_boxes, attrs


def _resolve_path_or_none(path: str | Path | None) -> Path | None:
    if not path:
        return None
    return Path(path).resolve()
