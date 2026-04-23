"""
Dataset statistics and distribution analysis (PRD §17).

Works on a list of annotation-record dicts (as produced by
`metadata_export.build_annotation_records`), so it is independent of
any GUI state.
"""
from __future__ import annotations

import csv
import io
import json
import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Sequence

from .attributes import compute_size_tag


def _bbox_dims(rec: Mapping[str, Any]) -> tuple[float, float, float, float]:
    """Return (width, height, area, aspect_ratio) from an annotation record."""
    x1, y1 = float(rec["x1"]), float(rec["y1"])
    x2, y2 = float(rec["x2"]), float(rec["y2"])
    w = abs(x2 - x1)
    h = abs(y2 - y1)
    area = w * h
    ar = (w / h) if h > 0 else 0.0
    return w, h, area, ar


def compute_dataset_stats(
    records: Sequence[Mapping[str, Any]],
    *,
    image_path_key: str = "image_path",
    total_images_override: int | None = None,
    labeled_images_override: int | None = None,
) -> Dict[str, Any]:
    """
    Compute all §17.1 statistics from annotation records.

    Returns a dict with keys:
        total_images, total_annotations,
        class_distribution, size_tag_distribution,
        bbox_width_distribution, bbox_height_distribution,
        bbox_area_distribution, aspect_ratio_distribution,
        class_x_size_distribution
    """
    image_set: set[str] = set()
    image_counter: Counter[str] = Counter()
    class_counter: Counter[str] = Counter()
    size_counter: Counter[str] = Counter()
    scene_counter: Counter[str] = Counter()
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    ars: list[float] = []
    rotation_angles: list[float] = []
    class_x_size: Counter[str] = Counter()

    for rec in records:
        img = rec.get(image_path_key, "") or ""
        if img:
            image_set.add(img)
            image_counter[img] += 1
        cls = str(rec.get("class_name", ""))
        class_counter[cls] += 1

        x1, y1 = float(rec["x1"]), float(rec["y1"])
        x2, y2 = float(rec["x2"]), float(rec["y2"])
        st = rec.get("size_tag") or compute_size_tag(x1, y1, x2, y2)
        size_counter[st] += 1
        class_x_size[f"{cls}|{st}"] += 1
        scene_tag = str(rec.get("scene_tag", "") or "unknown")
        scene_counter[scene_tag] += 1

        w, h, area, ar = _bbox_dims(rec)
        widths.append(w)
        heights.append(h)
        areas.append(area)
        ars.append(ar)
        angle = _extract_rotation_angle(rec)
        if angle is not None:
            rotation_angles.append(angle)

    labeled_images = (
        int(labeled_images_override)
        if labeled_images_override is not None
        else (len(image_set) if image_set else (1 if records else 0))
    )
    total_images = (
        int(total_images_override)
        if total_images_override is not None
        else (len(image_set) if image_set else (1 if records else 0))
    )
    total_annotations = len(records)
    image_annotation_values = list(image_counter.values()) if image_counter else ([total_annotations] if total_annotations else [])

    return {
        "total_images": total_images,
        "labeled_images": min(labeled_images, total_images) if total_images >= 0 else labeled_images,
        "unlabeled_images": max(total_images - labeled_images, 0),
        "total_annotations": total_annotations,
        "avg_annotations_per_image": round(total_annotations / total_images, 2) if total_images else 0.0,
        "class_distribution": dict(class_counter.most_common()),
        "size_tag_distribution": dict(size_counter.most_common()),
        "scene_tag_distribution": dict(scene_counter.most_common()),
        "scene_tag_ratio": _ratio_summary(scene_counter, total_annotations),
        "bbox_width_distribution": _numeric_summary(widths),
        "bbox_height_distribution": _numeric_summary(heights),
        "bbox_area_distribution": _numeric_summary(areas),
        "aspect_ratio_distribution": _numeric_summary(ars),
        "rotation_angle_distribution": _numeric_summary(rotation_angles),
        "image_annotation_distribution": _numeric_summary(image_annotation_values),
        "class_x_size_distribution": _class_x_size_table(class_x_size),
    }


def _numeric_summary(values: list[float]) -> Dict[str, float]:
    if not values:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "std": 0}
    n = len(values)
    mn = min(values)
    mx = max(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    return {
        "count": n,
        "min": round(mn, 2),
        "max": round(mx, 2),
        "mean": round(mean, 2),
        "std": round(math.sqrt(var), 2),
    }


def _ratio_summary(counter: Counter[str], total: int) -> Dict[str, float]:
    if total <= 0:
        return {}
    return {key: round((cnt / total) * 100.0, 2) for key, cnt in counter.most_common()}


def _extract_rotation_angle(rec: Mapping[str, Any]) -> float | None:
    for key in ("rotation_deg", "angle_deg", "rotation_angle_deg", "obb_angle_deg"):
        raw = rec.get(key)
        if raw in (None, ""):
            continue
        return float(raw)
    return None


def _class_x_size_table(
    counter: Counter[str],
) -> Dict[str, Dict[str, int]]:
    """Convert 'cls|size_tag' counter to nested {cls: {size_tag: count}}."""
    table: Dict[str, Dict[str, int]] = {}
    for key, cnt in counter.items():
        cls, st = key.split("|", 1)
        table.setdefault(cls, {})
        table[cls][st] = cnt
    return table


# --------------- export helpers ---------------

def export_stats_json(stats: Mapping[str, Any]) -> str:
    return json.dumps(dict(stats), indent=2, ensure_ascii=False) + "\n"


_SUMMARY_CSV_FIELDS = [
    "metric", "count", "min", "max", "mean", "std",
]


def export_stats_csv(stats: Mapping[str, Any]) -> str:
    """
    Flat CSV with one row per numeric summary + header rows for counts.
    """
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["section", "key", "value"])

    w.writerow(["overview", "total_images", stats.get("total_images", 0)])
    w.writerow(["overview", "labeled_images", stats.get("labeled_images", 0)])
    w.writerow(["overview", "unlabeled_images", stats.get("unlabeled_images", 0)])
    w.writerow(["overview", "total_annotations", stats.get("total_annotations", 0)])
    w.writerow(["overview", "avg_annotations_per_image", stats.get("avg_annotations_per_image", 0)])

    for cls, cnt in stats.get("class_distribution", {}).items():
        w.writerow(["class_distribution", cls, cnt])

    for st, cnt in stats.get("size_tag_distribution", {}).items():
        w.writerow(["size_tag_distribution", st, cnt])

    for scene, cnt in stats.get("scene_tag_distribution", {}).items():
        w.writerow(["scene_tag_distribution", scene, cnt])

    for scene, ratio in stats.get("scene_tag_ratio", {}).items():
        w.writerow(["scene_tag_ratio", scene, ratio])

    for metric in (
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "aspect_ratio",
        "rotation_angle",
        "image_annotation",
    ):
        dist = stats.get(f"{metric}_distribution", {})
        for k, v in dist.items():
            w.writerow([f"{metric}_distribution", k, v])

    cxs = stats.get("class_x_size_distribution", {})
    for cls, sizes in cxs.items():
        for st, cnt in sizes.items():
            w.writerow(["class_x_size", f"{cls}|{st}", cnt])

    return buf.getvalue()
