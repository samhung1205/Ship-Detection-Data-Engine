"""
Model prediction boxes for overlay (PRD Phase 2A). Origin-pixel HBB + confidence + status.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import List, Sequence


# pred_status (PRD §10.7)
STATUS_PREDICTED = "predicted"
STATUS_ACCEPTED = "accepted"
STATUS_EDITED = "edited"
STATUS_REJECTED = "rejected"


@dataclass
class PredictionRecord:
    pred_id: str
    class_id: int
    class_name: str
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    pred_status: str = STATUS_PREDICTED

    @classmethod
    def from_yolo_line(
        cls,
        parts: Sequence[str],
        *,
        object_list: Sequence[str],
        image_w: int,
        image_h: int,
    ) -> "PredictionRecord":
        """
        One line: class_id x_center y_center width height [confidence]
        Normalized YOLO HBB; optional 6th field confidence in [0,1].
        """
        if len(parts) < 5:
            raise ValueError("YOLO prediction line needs at least 5 fields")
        cid = int(float(parts[0]))
        xc, yc, nw, nh = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
        conf = float(parts[5]) if len(parts) >= 6 else 1.0

        if cid < 0 or cid >= len(object_list):
            raise IndexError(f"class_id {cid} out of range for object_list len {len(object_list)}")

        x1 = xc * image_w - (nw * image_w) / 2.0
        y1 = yc * image_h - (nh * image_h) / 2.0
        x2 = xc * image_w + (nw * image_w) / 2.0
        y2 = yc * image_h + (nh * image_h) / 2.0
        name = object_list[cid]

        return cls(
            pred_id=str(uuid.uuid4())[:12],
            class_id=cid,
            class_name=name,
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            confidence=conf,
            pred_status=STATUS_PREDICTED,
        )


def parse_predictions_yolo_txt(
    content: str,
    *,
    object_list: Sequence[str],
    image_w: int,
    image_h: int,
) -> List[PredictionRecord]:
    out: List[PredictionRecord] = []
    for raw in content.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        out.append(
            PredictionRecord.from_yolo_line(
                parts, object_list=object_list, image_w=image_w, image_h=image_h
            )
        )
    return out
