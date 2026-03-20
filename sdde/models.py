from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class HBBBoxPx:
    """
    Axis-aligned horizontal bounding box in pixel coordinates.

    Canonical form:
      - x1 <= x2
      - y1 <= y2
    """

    x1: float
    y1: float
    x2: float
    y2: float

    def __post_init__(self) -> None:
        x1, x2 = sorted([float(self.x1), float(self.x2)])
        y1, y2 = sorted([float(self.y1), float(self.y2)])
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @property
    def width_px(self) -> float:
        return self.x2 - self.x1

    @property
    def height_px(self) -> float:
        return self.y2 - self.y1

    def to_yolo_norm(self, image_w: int, image_h: int) -> "HBBBoxYoloNorm":
        if image_w <= 0 or image_h <= 0:
            raise ValueError("image_w/image_h must be positive")

        cx = (self.x1 + self.x2) / 2.0 / image_w
        cy = (self.y1 + self.y2) / 2.0 / image_h
        w = self.width_px / image_w
        h = self.height_px / image_h

        # Do not clamp here; keep the model honest. Import/export layer
        # can decide whether to clamp or reject out-of-range values.
        return HBBBoxYoloNorm(x_center=cx, y_center=cy, width=w, height=h)


@dataclass(frozen=True)
class HBBBoxYoloNorm:
    """
    Axis-aligned HBB in YOLO normalized form (center-based):
      - x_center, y_center: [0, 1] typically
      - width, height: relative size (usually [0, 1])
    """

    x_center: float
    y_center: float
    width: float
    height: float

    def to_px(self, image_w: int, image_h: int) -> HBBBoxPx:
        if image_w <= 0 or image_h <= 0:
            raise ValueError("image_w/image_h must be positive")

        w = self.width * image_w
        h = self.height * image_h
        cx = self.x_center * image_w
        cy = self.y_center * image_h

        x1 = cx - w / 2.0
        y1 = cy - h / 2.0
        x2 = cx + w / 2.0
        y2 = cy + h / 2.0
        return HBBBoxPx(x1=x1, y1=y1, x2=x2, y2=y2)


@dataclass
class ClassMapping:
    """
    Maps between class ids and class names.

    Notes:
    - The SDDE defaults are expected to match the current dataset mapping.
    - This model keeps the canonical mapping as names list order.
    """

    names: List[str] = field(default_factory=list)

    @classmethod
    def default_ship_mapping(cls) -> "ClassMapping":
        # Must match prompt.md defaults.
        return cls(
            names=["Naval", "Merchant", "Dock", "other_vessel"],
        )

    @property
    def nc(self) -> int:
        return len(self.names)

    def validate(self) -> None:
        if not self.names:
            raise ValueError("ClassMapping.names cannot be empty")
        if len(set(self.names)) != len(self.names):
            raise ValueError("ClassMapping.names must be unique")

    def id_to_name(self, class_id: int) -> str:
        if class_id < 0 or class_id >= self.nc:
            raise IndexError("class_id out of range")
        return self.names[class_id]

    def name_to_id(self, name: str) -> int:
        self.validate()
        return self.names.index(name)


@dataclass
class HBBAnnotation:
    """
    One HBB instance attached to an image.
    """

    class_id: int
    bbox_px: HBBBoxPx
    attributes: Dict[str, str] = field(default_factory=dict)

    def with_attribute(self, key: str, value: str) -> "HBBAnnotation":
        # Small convenience for UI/controller layer.
        self.attributes[key] = value
        return self

    def get_attribute(self, key: str, default: Optional[str] = None) -> Optional[str]:
        return self.attributes.get(key, default)


@dataclass
class ImageAnnotation:
    """
    Annotation data for a single image.
    """

    image_path: Optional[str] = None
    image_width_px: Optional[int] = None
    image_height_px: Optional[int] = None
    annotations: List[HBBAnnotation] = field(default_factory=list)

    def set_image_size(self, width_px: int, height_px: int) -> None:
        if width_px <= 0 or height_px <= 0:
            raise ValueError("image_width_px/image_height_px must be positive")
        self.image_width_px = int(width_px)
        self.image_height_px = int(height_px)

    def add_annotation(self, ann: HBBAnnotation) -> None:
        self.annotations.append(ann)

    def to_dict(self) -> dict:
        # Generic, UI-agnostic serialization hook.
        return {
            "image_path": self.image_path,
            "image_width_px": self.image_width_px,
            "image_height_px": self.image_height_px,
            "annotations": [
                {
                    "class_id": a.class_id,
                    "bbox_px": {"x1": a.bbox_px.x1, "y1": a.bbox_px.y1, "x2": a.bbox_px.x2, "y2": a.bbox_px.y2},
                    "attributes": dict(a.attributes),
                }
                for a in self.annotations
            ],
        }


# Backward-compatible alias name (prompt uses "annotation data")
AnnotationAttributes = Dict[str, str]

