from __future__ import annotations

import json
from pathlib import Path
from typing import List, Literal, Sequence
from xml.etree import ElementTree as ET

from .attributes import normalize_attributes
from .models import ClassMapping, HBBAnnotation, HBBBoxPx, HBBBoxYoloNorm


_ClsMode = Literal["class_id", "class_name"]
_BBoxOrder = Literal["class_first", "coords_first"]


def _format_number(v: float) -> str:
    """
    Format numbers for txt export.

    - If v is very close to an int, export as int (matches legacy UI behavior).
    - Otherwise export with fixed decimals (enough for YOLO normalized coords).
    """
    vi = round(v)
    if abs(v - vi) < 1e-6:
        return str(int(vi))
    s = f"{v:.6f}"
    s = s.rstrip("0").rstrip(".")
    return s


def parse_yolo_hbb_txt(
    content: str,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
) -> List[HBBAnnotation]:
    """
    Parse YOLO HBB label text into annotation objects.

    Expected line format:
      cls x_center y_center width height
    where coords are normalized to [0, 1] relative to image size.
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    annotations: List[HBBAnnotation] = []
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid YOLO line (need 5 fields): {raw_line!r}")

        cls_id = int(float(parts[0]))
        if cls_id < 0 or cls_id >= class_mapping.nc:
            raise IndexError(f"cls_id out of range: {cls_id}")

        x_center = float(parts[1])
        y_center = float(parts[2])
        w = float(parts[3])
        h = float(parts[4])

        yolo = HBBBoxYoloNorm(x_center=x_center, y_center=y_center, width=w, height=h)
        bbox_px = yolo.to_px(image_w=image_w, image_h=image_h)
        annotations.append(HBBAnnotation(class_id=cls_id, bbox_px=bbox_px))

    return annotations


def import_yolo_hbb_label_file(
    path: str | Path,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
) -> List[HBBAnnotation]:
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return parse_yolo_hbb_txt(content, class_mapping=class_mapping, image_w=image_w, image_h=image_h)


def parse_coco_bbox_json(
    content: str,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> List[HBBAnnotation]:
    """
    Parse a COCO bbox JSON payload and return annotations for the current image.

    Supports both:
    - single-image COCO JSON exported by this project
    - multi-image COCO datasets, matched by current image filename when possible
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    payload = json.loads(content)
    images = payload.get("images")
    annotations = payload.get("annotations")
    categories = payload.get("categories")
    if not isinstance(images, list) or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError("COCO JSON must contain images, annotations, and categories lists")

    target_image = _select_coco_image_record(images, image_path=image_path)
    if target_image is None:
        raise ValueError("Could not find a matching image entry in COCO JSON")

    target_image_id = int(target_image.get("id"))
    json_w = int(target_image.get("width", image_w) or image_w)
    json_h = int(target_image.get("height", image_h) or image_h)
    if json_w != image_w or json_h != image_h:
        raise ValueError(
            f"COCO image size {json_w}x{json_h} does not match current image {image_w}x{image_h}"
        )

    category_id_to_class_id = _build_coco_category_mapping(categories, class_mapping=class_mapping)
    imported: List[HBBAnnotation] = []
    for ann in annotations:
        if int(ann.get("image_id", -1)) != target_image_id:
            continue
        bbox = ann.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            raise ValueError("COCO annotation bbox must be [x, y, w, h]")
        cat_id = int(ann.get("category_id"))
        if cat_id not in category_id_to_class_id:
            raise ValueError(f"Unknown COCO category_id: {cat_id}")
        x, y, w, h = [float(v) for v in bbox[:4]]
        attrs = {}
        if int(ann.get("iscrowd", 0)):
            attrs["crowded"] = "true"
        imported.append(
            HBBAnnotation(
                class_id=category_id_to_class_id[cat_id],
                bbox_px=HBBBoxPx(x1=x, y1=y, x2=x + w, y2=y + h),
                attributes=attrs,
            )
        )
    return imported


def import_coco_bbox_json_file(
    path: str | Path,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> List[HBBAnnotation]:
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return parse_coco_bbox_json(
        content,
        class_mapping=class_mapping,
        image_w=image_w,
        image_h=image_h,
        image_path=image_path,
    )


def parse_annotation_metadata_json(
    content: str,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> List[HBBAnnotation]:
    """
    Parse exported annotation metadata JSON records back into annotations.
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    payload = json.loads(content)
    if not isinstance(payload, list):
        raise ValueError("Annotation metadata JSON must be a list of records")
    if payload and not all(isinstance(rec, dict) for rec in payload):
        raise ValueError("Annotation metadata JSON records must be objects")

    records = _select_metadata_records(payload, image_path=image_path)
    if not records:
        raise ValueError("Could not find matching annotation metadata records for the current image")

    imported: List[HBBAnnotation] = []
    for rec in records:
        rec_w = rec.get("image_width")
        rec_h = rec.get("image_height")
        if rec_w not in (None, "") and int(rec_w) != image_w:
            raise ValueError(f"Metadata image_width {rec_w} does not match current image {image_w}")
        if rec_h not in (None, "") and int(rec_h) != image_h:
            raise ValueError(f"Metadata image_height {rec_h} does not match current image {image_h}")

        class_name = str(rec.get("class_name", "")).strip()
        class_id = rec.get("class_id")
        if class_name and class_name in class_mapping.names:
            mapped_class_id = class_mapping.name_to_id(class_name)
        elif class_id not in (None, "") and 0 <= int(class_id) < class_mapping.nc:
            mapped_class_id = int(class_id)
        else:
            raise ValueError(
                f"Metadata record class cannot be mapped to current classes.yaml: {class_name or class_id}"
            )

        imported.append(
            HBBAnnotation(
                class_id=mapped_class_id,
                bbox_px=HBBBoxPx(
                    x1=float(rec["x1"]),
                    y1=float(rec["y1"]),
                    x2=float(rec["x2"]),
                    y2=float(rec["y2"]),
                ),
                attributes=normalize_attributes(rec),
            )
        )
    return imported


def import_annotation_metadata_json_file(
    path: str | Path,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> List[HBBAnnotation]:
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    return parse_annotation_metadata_json(
        content,
        class_mapping=class_mapping,
        image_w=image_w,
        image_h=image_h,
        image_path=image_path,
    )


def import_json_label_file(
    path: str | Path,
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> List[HBBAnnotation]:
    p = Path(path)
    content = p.read_text(encoding="utf-8")
    errors: list[str] = []
    try:
        return parse_coco_bbox_json(
            content,
            class_mapping=class_mapping,
            image_w=image_w,
            image_h=image_h,
            image_path=image_path,
        )
    except (ValueError, TypeError, KeyError, IndexError) as e:
        errors.append(f"COCO: {e}")
    try:
        return parse_annotation_metadata_json(
            content,
            class_mapping=class_mapping,
            image_w=image_w,
            image_h=image_h,
            image_path=image_path,
        )
    except (ValueError, TypeError, KeyError, IndexError) as e:
        errors.append(f"Metadata: {e}")
    raise ValueError("Unsupported JSON label format. " + " | ".join(errors))


def _select_coco_image_record(
    images: Sequence[dict],
    *,
    image_path: str | Path | None = None,
) -> dict | None:
    if len(images) == 1:
        image = images[0]
        return image if isinstance(image, dict) else None

    current_name = Path(image_path).name if image_path else ""
    if current_name:
        exact = next(
            (
                image for image in images
                if isinstance(image, dict)
                and Path(str(image.get("file_name", ""))).name == current_name
            ),
            None,
        )
        if exact is not None:
            return exact
    return None


def _select_metadata_records(
    records: Sequence[dict],
    *,
    image_path: str | Path | None = None,
) -> list[dict]:
    if not records:
        return []
    current_name = Path(image_path).name if image_path else ""
    if current_name:
        matched = [
            rec for rec in records
            if Path(str(rec.get("image_path", ""))).name == current_name
        ]
        if matched:
            return matched
    non_empty_names = {
        Path(str(rec.get("image_path", ""))).name
        for rec in records
        if str(rec.get("image_path", "")).strip()
    }
    if not non_empty_names:
        return list(records)
    if len(non_empty_names) == 1 and (not current_name or current_name in non_empty_names):
        return list(records)
    return []


def _build_coco_category_mapping(
    categories: Sequence[dict],
    *,
    class_mapping: ClassMapping,
) -> dict[int, int]:
    category_id_to_class_id: dict[int, int] = {}
    for cat in categories:
        if not isinstance(cat, dict):
            raise ValueError("COCO categories must be objects")
        cat_id = int(cat.get("id"))
        cat_name = str(cat.get("name", "")).strip()
        if cat_name in class_mapping.names:
            category_id_to_class_id[cat_id] = class_mapping.name_to_id(cat_name)
            continue
        if 0 <= cat_id < class_mapping.nc:
            category_id_to_class_id[cat_id] = cat_id
            continue
        raise ValueError(f"COCO category cannot be mapped to current classes.yaml: {cat_name or cat_id}")
    return category_id_to_class_id


def export_yolo_hbb_txt(
    annotations: Sequence[HBBAnnotation],
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    include_trailing_newline: bool = True,
) -> str:
    """
    Export annotations into YOLO HBB txt content:
      cls x_center y_center width height
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    lines: List[str] = []
    for ann in annotations:
        cls_id = ann.class_id
        if cls_id < 0 or cls_id >= class_mapping.nc:
            raise IndexError(f"ann.class_id out of range: {cls_id}")

        yolo = ann.bbox_px.to_yolo_norm(image_w=image_w, image_h=image_h)
        lines.append(
            f"{cls_id} "
            f"{_format_number(yolo.x_center)} {_format_number(yolo.y_center)} "
            f"{_format_number(yolo.width)} {_format_number(yolo.height)}"
        )

    out = "\n".join(lines)
    if include_trailing_newline and out:
        out += "\n"
    return out


def export_bbox_txt(
    annotations: Sequence[HBBAnnotation],
    *,
    class_mapping: ClassMapping,
    cls_mode: _ClsMode = "class_id",
    order: _BBoxOrder = "class_first",
    include_trailing_newline: bool = True,
) -> str:
    """
    Export absolute bbox coordinates as txt.

    Format:
      - ``class_first``: ``cls x1 y1 x2 y2``
      - ``coords_first``: ``x1 y1 x2 y2 cls``
    where (x1,y1) is top-left and (x2,y2) is bottom-right in px.
    """
    class_mapping.validate()
    if cls_mode not in ("class_id", "class_name"):
        raise ValueError("cls_mode must be 'class_id' or 'class_name'")
    if order not in ("class_first", "coords_first"):
        raise ValueError("order must be 'class_first' or 'coords_first'")

    lines: List[str] = []
    for ann in annotations:
        cls_id = ann.class_id
        if cls_id < 0 or cls_id >= class_mapping.nc:
            raise IndexError(f"ann.class_id out of range: {cls_id}")

        cls_val: str
        if cls_mode == "class_id":
            cls_val = str(cls_id)
        else:
            cls_val = class_mapping.id_to_name(cls_id)

        x1 = ann.bbox_px.x1
        y1 = ann.bbox_px.y1
        x2 = ann.bbox_px.x2
        y2 = ann.bbox_px.y2

        if order == "class_first":
            line = (
                f"{cls_val} "
                f"{_format_number(x1)} {_format_number(y1)} "
                f"{_format_number(x2)} {_format_number(y2)}"
            )
        else:
            line = (
                f"{_format_number(x1)} {_format_number(y1)} "
                f"{_format_number(x2)} {_format_number(y2)} {cls_val}"
            )

        lines.append(line)

    out = "\n".join(lines)
    if include_trailing_newline and out:
        out += "\n"
    return out


def export_coco_bbox_json(
    annotations: Sequence[HBBAnnotation],
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
    image_id: int = 1,
) -> str:
    """
    Export one image's HBB annotations as a COCO-style bbox JSON document.
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    file_name = Path(image_path).name if image_path else ""
    payload = {
        "images": [
            {
                "id": image_id,
                "file_name": file_name,
                "width": int(image_w),
                "height": int(image_h),
            }
        ],
        "annotations": [],
        "categories": [
            {
                "id": class_id,
                "name": class_mapping.id_to_name(class_id),
                "supercategory": "",
            }
            for class_id in range(class_mapping.nc)
        ],
    }

    coco_annotations: List[dict] = []
    for ann_id, ann in enumerate(annotations, start=1):
        cls_id = ann.class_id
        if cls_id < 0 or cls_id >= class_mapping.nc:
            raise IndexError(f"ann.class_id out of range: {cls_id}")

        bbox = ann.bbox_px
        crowded = str(ann.get_attribute("crowded", "false")).lower() == "true"
        coco_annotations.append(
            {
                "id": ann_id,
                "image_id": image_id,
                "category_id": cls_id,
                "bbox": [
                    float(bbox.x1),
                    float(bbox.y1),
                    float(bbox.width_px),
                    float(bbox.height_px),
                ],
                "area": float(bbox.width_px * bbox.height_px),
                "iscrowd": int(crowded),
            }
        )
    payload["annotations"] = coco_annotations
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def export_pascal_voc_xml(
    annotations: Sequence[HBBAnnotation],
    *,
    class_mapping: ClassMapping,
    image_w: int,
    image_h: int,
    image_path: str | Path | None = None,
) -> str:
    """
    Export one image's HBB annotations as a Pascal VOC XML document.
    """
    class_mapping.validate()
    if image_w <= 0 or image_h <= 0:
        raise ValueError("image_w/image_h must be positive")

    image_path_obj = Path(image_path) if image_path else None
    root = ET.Element("annotation")
    ET.SubElement(root, "folder").text = image_path_obj.parent.name if image_path_obj else ""
    ET.SubElement(root, "filename").text = image_path_obj.name if image_path_obj else ""
    ET.SubElement(root, "path").text = str(image_path_obj) if image_path_obj else ""

    source = ET.SubElement(root, "source")
    ET.SubElement(source, "database").text = "Unknown"

    size = ET.SubElement(root, "size")
    ET.SubElement(size, "width").text = str(int(image_w))
    ET.SubElement(size, "height").text = str(int(image_h))
    ET.SubElement(size, "depth").text = "3"
    ET.SubElement(root, "segmented").text = "0"

    for ann in annotations:
        cls_id = ann.class_id
        if cls_id < 0 or cls_id >= class_mapping.nc:
            raise IndexError(f"ann.class_id out of range: {cls_id}")

        obj = ET.SubElement(root, "object")
        ET.SubElement(obj, "name").text = class_mapping.id_to_name(cls_id)
        ET.SubElement(obj, "pose").text = "Unspecified"
        ET.SubElement(obj, "truncated").text = (
            "1" if str(ann.get_attribute("truncated", "false")).lower() == "true" else "0"
        )
        ET.SubElement(obj, "difficult").text = (
            "1" if str(ann.get_attribute("hard_sample", "false")).lower() == "true" else "0"
        )

        bbox = ET.SubElement(obj, "bndbox")
        ET.SubElement(bbox, "xmin").text = str(int(round(ann.bbox_px.x1)))
        ET.SubElement(bbox, "ymin").text = str(int(round(ann.bbox_px.y1)))
        ET.SubElement(bbox, "xmax").text = str(int(round(ann.bbox_px.x2)))
        ET.SubElement(bbox, "ymax").text = str(int(round(ann.bbox_px.y2)))

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"
