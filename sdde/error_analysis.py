"""
Error analysis: GT-vs-prediction matching and error-type classification (PRD §14).

Computes IoU between GT and predicted HBBs, performs greedy matching, and
classifies each case as TP, FP, FN, WrongClass, Localization, or Duplicate.
"""
from __future__ import annotations

import csv
import io
import uuid
from dataclasses import dataclass, field
from typing import Sequence

from .prediction import PredictionRecord

# Error types (PRD §14.3)
ERROR_FP = "FP"
ERROR_FN = "FN"
ERROR_WRONG_CLASS = "WrongClass"
ERROR_LOCALIZATION = "Localization"
ERROR_DUPLICATE = "Duplicate"
ERROR_TP = "TP"

ALL_ERROR_TYPES = (ERROR_TP, ERROR_FP, ERROR_FN, ERROR_WRONG_CLASS, ERROR_LOCALIZATION, ERROR_DUPLICATE)


def iou_xyxy(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    """Intersection-over-union for two (x1, y1, x2, y2) boxes."""
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


@dataclass
class ErrorCase:
    """One GT-pred pair (or unmatched GT / pred) with analysis metadata (PRD §10.8)."""
    case_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    image_id: str = ""
    gt_index: int | None = None
    pred_index: int | None = None
    error_type: str = ""
    iou: float = 0.0
    gt_class: str = ""
    pred_class: str = ""
    confidence: float = 0.0
    notes: str = ""
    bookmarked: bool = False


def match_gt_pred(
    gt_boxes: Sequence[tuple[str, float, float, float, float]],
    predictions: Sequence[PredictionRecord],
    *,
    iou_threshold: float = 0.5,
    localization_low: float = 0.1,
    image_id: str = "",
) -> list[ErrorCase]:
    """
    Greedy IoU matching between GT and prediction boxes.

    Parameters
    ----------
    gt_boxes : sequence of (class_name, x1, y1, x2, y2) in origin-pixel space
    predictions : PredictionRecord list (origin-pixel coords, any pred_status)
    iou_threshold : IoU >= this → potential match (TP / WrongClass)
    localization_low : IoU in [localization_low, iou_threshold) → Localization error

    Returns
    -------
    List of ErrorCase covering every GT and every prediction exactly once.

    Classification rules
    --------------------
    1. Build IoU matrix; sort pairs descending by IoU.
    2. Greedy: pick highest IoU pair that has both GT and pred unmatched.
       - IoU >= threshold AND same class → TP
       - IoU >= threshold AND diff class → WrongClass
       - localization_low <= IoU < threshold → Localization
    3. Any pred matched to a GT that was already matched → Duplicate
    4. Remaining unmatched preds → FP
    5. Remaining unmatched GTs  → FN
    """
    n_gt = len(gt_boxes)
    n_pred = len(predictions)

    pairs: list[tuple[float, int, int]] = []
    for gi in range(n_gt):
        ga = (gt_boxes[gi][1], gt_boxes[gi][2], gt_boxes[gi][3], gt_boxes[gi][4])
        for pi in range(n_pred):
            p = predictions[pi]
            pa = (p.x1, p.y1, p.x2, p.y2)
            v = iou_xyxy(ga, pa)
            if v > 0:
                pairs.append((v, gi, pi))
    pairs.sort(key=lambda t: -t[0])

    gt_matched: dict[int, int] = {}
    pred_matched: dict[int, int] = {}
    cases: list[ErrorCase] = []

    for v, gi, pi in pairs:
        if gi in gt_matched and pi in pred_matched:
            continue
        if pi in pred_matched:
            continue

        if gi in gt_matched:
            # pred overlaps a GT that is already matched → Duplicate
            cases.append(ErrorCase(
                image_id=image_id,
                gt_index=gi,
                pred_index=pi,
                error_type=ERROR_DUPLICATE,
                iou=v,
                gt_class=gt_boxes[gi][0],
                pred_class=predictions[pi].class_name,
                confidence=predictions[pi].confidence,
            ))
            pred_matched[pi] = gi
            continue

        gt_cls = gt_boxes[gi][0]
        pred_cls = predictions[pi].class_name

        if v >= iou_threshold:
            if gt_cls == pred_cls:
                etype = ERROR_TP
            else:
                etype = ERROR_WRONG_CLASS
        elif v >= localization_low:
            etype = ERROR_LOCALIZATION
        else:
            continue

        cases.append(ErrorCase(
            image_id=image_id,
            gt_index=gi,
            pred_index=pi,
            error_type=etype,
            iou=v,
            gt_class=gt_cls,
            pred_class=pred_cls,
            confidence=predictions[pi].confidence,
        ))
        gt_matched[gi] = pi
        pred_matched[pi] = gi

    for pi in range(n_pred):
        if pi not in pred_matched:
            cases.append(ErrorCase(
                image_id=image_id,
                pred_index=pi,
                error_type=ERROR_FP,
                iou=0.0,
                pred_class=predictions[pi].class_name,
                confidence=predictions[pi].confidence,
            ))

    for gi in range(n_gt):
        if gi not in gt_matched:
            cases.append(ErrorCase(
                image_id=image_id,
                gt_index=gi,
                error_type=ERROR_FN,
                iou=0.0,
                gt_class=gt_boxes[gi][0],
            ))

    return cases


def summarise_error_cases(cases: Sequence[ErrorCase]) -> dict[str, int]:
    """Count occurrences of each error_type."""
    counts: dict[str, int] = {}
    for c in cases:
        counts[c.error_type] = counts.get(c.error_type, 0) + 1
    return counts


_CSV_FIELDS = [
    "case_id", "image_id", "error_type", "iou",
    "gt_index", "gt_class", "pred_index", "pred_class",
    "confidence", "notes", "bookmarked",
]


def export_error_cases_csv(cases: Sequence[ErrorCase]) -> str:
    """Serialise error cases to CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for c in cases:
        writer.writerow({
            "case_id": c.case_id,
            "image_id": c.image_id,
            "error_type": c.error_type,
            "iou": f"{c.iou:.4f}",
            "gt_index": c.gt_index if c.gt_index is not None else "",
            "gt_class": c.gt_class,
            "pred_index": c.pred_index if c.pred_index is not None else "",
            "pred_class": c.pred_class,
            "confidence": f"{c.confidence:.4f}",
            "notes": c.notes,
            "bookmarked": c.bookmarked,
        })
    return buf.getvalue()
