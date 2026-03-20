from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Sequence

from .models import ClassMapping, HBBAnnotation, HBBBoxPx, HBBBoxYoloNorm


_ClsMode = Literal["class_id", "class_name"]


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
    include_trailing_newline: bool = True,
) -> str:
    """
    Export absolute bbox coordinates as txt.

    Format:
      cls x1 y1 x2 y2
    where (x1,y1) is top-left and (x2,y2) is bottom-right in px.
    """
    class_mapping.validate()
    if cls_mode not in ("class_id", "class_name"):
        raise ValueError("cls_mode must be 'class_id' or 'class_name'")

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

        lines.append(
            f"{cls_val} {_format_number(x1)} {_format_number(y1)} {_format_number(x2)} {_format_number(y2)}"
        )

    out = "\n".join(lines)
    if include_trailing_newline and out:
        out += "\n"
    return out

