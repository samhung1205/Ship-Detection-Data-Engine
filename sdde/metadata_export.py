"""
Export per-image box annotations + attributes to JSON / CSV (research metadata).
"""
from __future__ import annotations

import csv
import io
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence


def _class_id_for_name(name: str, object_list: Sequence[str]) -> Optional[int]:
    try:
        return list(object_list).index(name)
    except ValueError:
        return None


def build_annotation_records(
    *,
    image_path: Optional[str],
    image_width: Optional[int],
    image_height: Optional[int],
    real_data: Sequence[Sequence[Any]],
    box_attributes: Sequence[Mapping[str, str]],
    object_list: Sequence[str],
    class_id_to_super: Optional[Mapping[int, str]] = None,
) -> List[Dict[str, Any]]:
    """
    real_data rows: [name, x1, y1, x2, y2]
    box_attributes[i]: size_tag, crowded, difficulty_tag, scene_tag
    """
    rows: List[Dict[str, Any]] = []
    for i, r in enumerate(real_data):
        name = str(r[0])
        x1, y1, x2, y2 = (float(r[1]), float(r[2]), float(r[3]), float(r[4]))
        cid = _class_id_for_name(name, object_list)
        attrs = dict(box_attributes[i]) if i < len(box_attributes) else {}
        rec: Dict[str, Any] = {
            "image_path": image_path,
            "image_width": image_width,
            "image_height": image_height,
            "class_name": name,
            "class_id": cid,
            "super_category": class_id_to_super.get(cid) if class_id_to_super and cid is not None else None,
            "x1": x1,
            "y1": y1,
            "x2": x2,
            "y2": y2,
            "size_tag": attrs.get("size_tag", "medium"),
            "crowded": attrs.get("crowded", "false"),
            "difficulty_tag": attrs.get("difficulty_tag", "normal"),
            "scene_tag": attrs.get("scene_tag", "unknown"),
        }
        rows.append(rec)
    return rows


def export_annotations_json(
    records: Sequence[Mapping[str, Any]], *, indent: int = 2
) -> str:
    return json.dumps(list(records), indent=indent, ensure_ascii=False) + "\n"


def export_annotations_csv(records: Sequence[Mapping[str, Any]]) -> str:
    if not records:
        return ""
    fieldnames = list(records[0].keys())
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for row in records:
        w.writerow({k: row.get(k, "") for k in fieldnames})
    return buf.getvalue()
