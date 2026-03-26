"""
Copy-paste augmentation metadata (PRD §16.3 / feature list §9.2).

Each paste operation is recorded as a PasteRecord so experiments are
reproducible: which asset, what transforms, where it landed, etc.
"""
from __future__ import annotations

import csv
import io
import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Sequence


@dataclass
class PasteRecord:
    """One copy-paste augmentation operation."""
    paste_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    image_path: str = ""
    asset_path: str = ""
    class_name: str = ""

    # transform params applied to the asset
    scale: float = 1.0
    rotation_deg: float = 0.0
    h_flip: bool = False
    brightness: int = 100
    contrast: int = 100

    # resulting bbox in origin-pixel space
    bbox_x1: int = 0
    bbox_y1: int = 0
    bbox_x2: int = 0
    bbox_y2: int = 0

    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(timespec="seconds")
    )


def bbox_from_legacy_paste_row(row: Sequence[object]) -> tuple[int, int, int, int]:
    """
    Extract origin-pixel bbox coords from a legacy paste row.

    Supported row shapes:
    - [x1, y1, x2, y2]
    - [class_name, x1, y1, x2, y2]
    """
    if len(row) < 4:
        raise ValueError("Paste bbox row must contain at least 4 values.")
    start = 1 if len(row) >= 5 else 0
    try:
        return (
            int(row[start]),
            int(row[start + 1]),
            int(row[start + 2]),
            int(row[start + 3]),
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Paste bbox row contains non-integer coordinates.") from exc


def export_paste_records_json(records: Sequence[PasteRecord]) -> str:
    """Serialize paste records to a JSON string."""
    return json.dumps([asdict(r) for r in records], indent=2, ensure_ascii=False)


_CSV_FIELDS = [
    "paste_id", "image_path", "asset_path", "class_name",
    "scale", "rotation_deg", "h_flip", "brightness", "contrast",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2", "timestamp",
]


def export_paste_records_csv(records: Sequence[PasteRecord]) -> str:
    """Serialize paste records to a CSV string."""
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=_CSV_FIELDS)
    writer.writeheader()
    for r in records:
        d = asdict(r)
        writer.writerow({k: d[k] for k in _CSV_FIELDS})
    return buf.getvalue()
