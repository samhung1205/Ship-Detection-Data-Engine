"""
Class mapping for SDDE: class_id, class_name, super_category (PRD §6).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple


@dataclass(frozen=True)
class ClassInfo:
    class_id: int
    class_name: str
    super_category: str


@dataclass(frozen=True)
class ClassCatalog:
    project_name: str
    classes: Tuple[ClassInfo, ...]

    @classmethod
    def from_list(cls, project_name: str, classes: Sequence[ClassInfo]) -> "ClassCatalog":
        return cls(project_name=project_name, classes=tuple(classes))

    def validate(self) -> None:
        if not self.classes:
            raise ValueError("ClassCatalog must contain at least one class")
        ids = [c.class_id for c in self.classes]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate class_id")
        names = [c.class_name for c in self.classes]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate class_name")
        sorted_ids = sorted(ids)
        if sorted_ids[0] != 0 or sorted_ids != list(range(len(sorted_ids))):
            raise ValueError("class_id must be contiguous integers 0 .. N-1")

    def names_ordered(self) -> List[str]:
        """YOLO / GUI list order: index i == class_id i."""
        return [c.class_name for c in sorted(self.classes, key=lambda c: c.class_id)]

    def signature(self) -> Tuple[Tuple[int, str, str], ...]:
        """Stable compare for 'mapping changed' warnings."""
        return tuple(
            (c.class_id, c.class_name, c.super_category)
            for c in sorted(self.classes, key=lambda c: c.class_id)
        )


def default_ship_catalog() -> ClassCatalog:
    """PRD default mapping (lowercase names)."""
    return ClassCatalog.from_list(
        "ship_detection_project",
        [
            ClassInfo(0, "naval", "vessel"),
            ClassInfo(1, "merchant", "vessel"),
            ClassInfo(2, "dock", "facility"),
            ClassInfo(3, "other_vessel", "vessel"),
        ],
    )
