"""
Dataset/folder annotation scanning helpers for analysis workflows.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct
from typing import Any, Mapping, Sequence

from .import_export import import_json_label_file, import_yolo_hbb_label_file
from .image_browser import SUPPORTED_IMAGE_SUFFIXES, list_supported_images
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
    recursive: bool = False,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
    current_image_path: str | Path | None = None,
    current_image_records: Sequence[Mapping[str, Any]] | None = None,
) -> FolderAnnotationScanResult:
    folder_path = Path(folder)
    image_paths = tuple(_list_image_paths(folder_path, recursive=recursive))
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
    inferred_base = _inferred_label_base(image_path)
    if inferred_base is not None:
        bases.append(inferred_base)
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


def _list_image_paths(folder: Path, *, recursive: bool) -> list[str]:
    if not recursive:
        return list_supported_images(folder)
    if not folder.is_dir():
        return []
    images = [
        str(path)
        for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    ]
    return sorted(images, key=lambda value: str(Path(value)).lower())


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


def _inferred_label_base(image_path: Path) -> Path | None:
    resolved = image_path.resolve()
    image_roots = {"images", "imgs", "image"}
    for ancestor in (resolved.parent, *resolved.parent.parents):
        if ancestor.name.lower() not in image_roots:
            continue
        try:
            rel = resolved.relative_to(ancestor)
        except ValueError:
            continue
        return ancestor.parent / "labels" / rel.parent / rel.stem
    return None


def read_image_size(path: str | Path) -> tuple[int, int] | None:
    image_path = Path(path)
    try:
        with image_path.open("rb") as f:
            header = f.read(256 * 1024)
    except OSError:
        return None
    if len(header) < 10:
        return None

    size = (
        _read_png_size(header)
        or _read_gif_size(header)
        or _read_bmp_size(header)
        or _read_jpeg_size(header)
        or _read_tiff_size(header)
    )
    return size


def _read_png_size(header: bytes) -> tuple[int, int] | None:
    if len(header) < 24 or header[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    if header[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", header[16:24])
    return int(width), int(height)


def _read_gif_size(header: bytes) -> tuple[int, int] | None:
    if len(header) < 10 or header[:6] not in (b"GIF87a", b"GIF89a"):
        return None
    width, height = struct.unpack("<HH", header[6:10])
    return int(width), int(height)


def _read_bmp_size(header: bytes) -> tuple[int, int] | None:
    if len(header) < 26 or header[:2] != b"BM":
        return None
    dib_header_size = struct.unpack("<I", header[14:18])[0]
    if dib_header_size < 12:
        return None
    if dib_header_size == 12:
        width, height = struct.unpack("<HH", header[18:22])
    else:
        width, height = struct.unpack("<ii", header[18:26])
        height = abs(height)
    return int(width), int(height)


def _read_jpeg_size(header: bytes) -> tuple[int, int] | None:
    if len(header) < 4 or header[:2] != b"\xff\xd8":
        return None
    i = 2
    while i + 9 < len(header):
        if header[i] != 0xFF:
            i += 1
            continue
        marker = header[i + 1]
        i += 2
        if marker in {0xD8, 0xD9}:
            continue
        if i + 2 > len(header):
            return None
        segment_length = struct.unpack(">H", header[i:i + 2])[0]
        if segment_length < 2:
            return None
        if marker in {
            0xC0, 0xC1, 0xC2, 0xC3,
            0xC5, 0xC6, 0xC7,
            0xC9, 0xCA, 0xCB,
            0xCD, 0xCE, 0xCF,
        }:
            if i + 7 > len(header):
                return None
            height, width = struct.unpack(">HH", header[i + 3:i + 7])
            return int(width), int(height)
        i += segment_length
    return None


def _read_tiff_size(header: bytes) -> tuple[int, int] | None:
    if len(header) < 8:
        return None
    byte_order = header[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None
    magic = struct.unpack(f"{endian}H", header[2:4])[0]
    if magic != 42:
        return None
    ifd_offset = struct.unpack(f"{endian}I", header[4:8])[0]
    if ifd_offset + 2 > len(header):
        return None
    entry_count = struct.unpack(f"{endian}H", header[ifd_offset:ifd_offset + 2])[0]
    width = None
    height = None
    entry_start = ifd_offset + 2
    for idx in range(entry_count):
        offset = entry_start + idx * 12
        if offset + 12 > len(header):
            break
        tag, field_type, count, value_or_offset = struct.unpack(
            f"{endian}HHII", header[offset:offset + 12]
        )
        if tag not in {256, 257}:
            continue
        value = _tiff_entry_value(
            header,
            endian=endian,
            field_type=field_type,
            count=count,
            value_or_offset=value_or_offset,
        )
        if value is None:
            continue
        if tag == 256:
            width = value
        elif tag == 257:
            height = value
        if width is not None and height is not None:
            return int(width), int(height)
    return None


def _tiff_entry_value(
    header: bytes,
    *,
    endian: str,
    field_type: int,
    count: int,
    value_or_offset: int,
) -> int | None:
    if count != 1:
        return None
    if field_type == 3:
        if endian == "<":
            return int(value_or_offset & 0xFFFF)
        return int(value_or_offset >> 16)
    if field_type == 4:
        return int(value_or_offset)
    value_sizes = {3: 2, 4: 4}
    size = value_sizes.get(field_type)
    if size is None:
        return None
    if value_or_offset + size > len(header):
        return None
    value_bytes = header[value_or_offset:value_or_offset + size]
    if field_type == 3:
        return int(struct.unpack(f"{endian}H", value_bytes)[0])
    if field_type == 4:
        return int(struct.unpack(f"{endian}I", value_bytes)[0])
    return None


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
