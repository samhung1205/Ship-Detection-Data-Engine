"""
Dataset / project validation helpers for research QC workflows.
"""
from __future__ import annotations

import csv
import io
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .dataset_scan import _candidate_label_paths, _list_image_paths


@dataclass(frozen=True)
class ValidationIssue:
    image_path: str
    file_path: str
    source: str
    issue_type: str
    detail: str
    line_no: int = 0


@dataclass(frozen=True)
class DatasetValidationResult:
    scope_path: str
    image_paths: tuple[str, ...]
    matched_label_paths: tuple[str, ...]
    matched_prediction_paths: tuple[str, ...]
    issues: tuple[ValidationIssue, ...]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def matched_labels(self) -> int:
        return len(self.matched_label_paths)

    @property
    def matched_predictions(self) -> int:
        return len(self.matched_prediction_paths)

    @property
    def total_issues(self) -> int:
        return len(self.issues)

    @property
    def images_with_issues(self) -> int:
        return len({issue.image_path for issue in self.issues})

    @property
    def clean_images(self) -> int:
        return max(0, self.total_images - self.images_with_issues)

    @property
    def issue_type_counts(self) -> dict[str, int]:
        return dict(Counter(issue.issue_type for issue in self.issues))

    @property
    def source_issue_counts(self) -> dict[str, int]:
        return dict(Counter(issue.source for issue in self.issues))

    @property
    def per_image_issue_counts(self) -> dict[str, int]:
        return dict(Counter(issue.image_path for issue in self.issues))


def scan_dataset_validation(
    scope_path: str | Path,
    *,
    object_list: Sequence[str],
    recursive: bool = False,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
    prediction_root: str | Path | None = None,
) -> DatasetValidationResult:
    root = Path(scope_path)
    image_paths = tuple(_list_image_paths(root, recursive=recursive))
    matched_label_paths: list[str] = []
    matched_prediction_paths: list[str] = []
    issues: list[ValidationIssue] = []

    for image_path_str in image_paths:
        image_path = Path(image_path_str).resolve()
        label_path = _first_existing_label_path(
            image_path,
            image_root=image_root,
            label_root=label_root,
        )
        if label_path is None:
            issues.append(
                ValidationIssue(
                    image_path=str(image_path),
                    file_path="",
                    source="label",
                    issue_type="missing_label",
                    detail="No matching label sidecar found.",
                )
            )
        else:
            matched_label_paths.append(str(label_path))
            issues.extend(
                _validate_label_file(
                    image_path=image_path,
                    label_path=label_path,
                    object_list=object_list,
                )
            )

        if prediction_root is not None:
            pred_path = _prediction_path_for_image(
                image_path,
                prediction_root=Path(prediction_root),
                image_root=image_root,
            )
            if not pred_path.is_file():
                issues.append(
                    ValidationIssue(
                        image_path=str(image_path),
                        file_path=str(pred_path),
                        source="prediction",
                        issue_type="missing_prediction",
                        detail="No matching prediction sidecar found.",
                    )
                )
            else:
                matched_prediction_paths.append(str(pred_path))
                issues.extend(
                    _validate_prediction_file(
                        image_path=image_path,
                        prediction_path=pred_path,
                        object_list=object_list,
                    )
                )

    return DatasetValidationResult(
        scope_path=str(root),
        image_paths=image_paths,
        matched_label_paths=tuple(matched_label_paths),
        matched_prediction_paths=tuple(matched_prediction_paths),
        issues=tuple(issues),
    )


def export_validation_issues_csv(result: DatasetValidationResult) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(["image_path", "file_path", "source", "issue_type", "line_no", "detail"])
    for issue in result.issues:
        writer.writerow([
            issue.image_path,
            issue.file_path,
            issue.source,
            issue.issue_type,
            issue.line_no,
            issue.detail,
        ])
    return out.getvalue()


def export_validation_summary_json(result: DatasetValidationResult) -> str:
    return json.dumps(build_validation_summary(result), ensure_ascii=False, indent=2)


def build_validation_summary(result: DatasetValidationResult) -> dict[str, Any]:
    return {
        "scope_path": result.scope_path,
        "total_images": result.total_images,
        "matched_labels": result.matched_labels,
        "matched_predictions": result.matched_predictions,
        "clean_images": result.clean_images,
        "images_with_issues": result.images_with_issues,
        "total_issues": result.total_issues,
        "issue_type_counts": result.issue_type_counts,
        "source_issue_counts": result.source_issue_counts,
        "per_image_issue_counts": result.per_image_issue_counts,
    }


def _first_existing_label_path(
    image_path: Path,
    *,
    image_root: str | Path | None,
    label_root: str | Path | None,
) -> Path | None:
    for candidate in _candidate_label_paths(
        image_path,
        image_root=image_root,
        label_root=label_root,
    ):
        if candidate.is_file():
            return candidate
    return None


def _prediction_path_for_image(
    image_path: Path,
    *,
    prediction_root: Path,
    image_root: str | Path | None,
) -> Path:
    if image_root:
        try:
            rel = image_path.resolve().relative_to(Path(image_root).resolve())
        except ValueError:
            rel = None
        if rel is not None:
            return prediction_root / rel.parent / f"{image_path.stem}.txt"
    return prediction_root / f"{image_path.stem}.txt"


def _validate_label_file(
    *,
    image_path: Path,
    label_path: Path,
    object_list: Sequence[str],
) -> list[ValidationIssue]:
    suffix = label_path.suffix.lower()
    if suffix == ".txt":
        return _validate_yolo_file(
            image_path=image_path,
            file_path=label_path,
            object_list=object_list,
            source="label",
            allow_confidence=False,
        )
    if suffix == ".json":
        try:
            payload = json.loads(label_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return [_issue(image_path, label_path, "label", "invalid_label_json", str(exc))]
        if isinstance(payload, list) and len(payload) == 0:
            return [_issue(image_path, label_path, "label", "empty_label", "JSON label file contains no records.")]
        return []
    return [_issue(image_path, label_path, "label", "unsupported_label_format", f"Unsupported suffix: {suffix}")]


def _validate_prediction_file(
    *,
    image_path: Path,
    prediction_path: Path,
    object_list: Sequence[str],
) -> list[ValidationIssue]:
    return _validate_yolo_file(
        image_path=image_path,
        file_path=prediction_path,
        object_list=object_list,
        source="prediction",
        allow_confidence=True,
    )


def _validate_yolo_file(
    *,
    image_path: Path,
    file_path: Path,
    object_list: Sequence[str],
    source: str,
    allow_confidence: bool,
) -> list[ValidationIssue]:
    try:
        content = file_path.read_text(encoding="utf-8")
    except OSError as exc:
        return [_issue(image_path, file_path, source, f"unreadable_{source}", str(exc))]

    lines = [line.strip() for line in content.splitlines() if line.strip() and not line.strip().startswith("#")]
    if not lines:
        return [_issue(image_path, file_path, source, f"empty_{source}", f"{source.title()} file contains no rows.")]

    issues: list[ValidationIssue] = []
    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        min_fields = 6 if allow_confidence else 5
        if len(parts) < min_fields:
            issues.append(
                _issue(
                    image_path,
                    file_path,
                    source,
                    f"invalid_{source}_line",
                    f"Expected at least {min_fields} fields, got {len(parts)}.",
                    line_no=idx,
                )
            )
            continue
        try:
            class_id = int(float(parts[0]))
            xc = float(parts[1])
            yc = float(parts[2])
            w = float(parts[3])
            h = float(parts[4])
            conf = float(parts[5]) if allow_confidence else None
        except ValueError:
            issues.append(
                _issue(
                    image_path,
                    file_path,
                    source,
                    f"invalid_{source}_line",
                    "Could not parse numeric YOLO fields.",
                    line_no=idx,
                )
            )
            continue
        if class_id < 0 or class_id >= len(object_list):
            issues.append(
                _issue(
                    image_path,
                    file_path,
                    source,
                    f"invalid_{source}_class_id",
                    f"class_id {class_id} is outside current class range 0..{len(object_list) - 1}.",
                    line_no=idx,
                )
            )
        issues.extend(
            _validate_normalized_bbox(
                image_path=image_path,
                file_path=file_path,
                source=source,
                line_no=idx,
                xc=xc,
                yc=yc,
                w=w,
                h=h,
            )
        )
        if conf is not None and not (0.0 <= conf <= 1.0):
            issues.append(
                _issue(
                    image_path,
                    file_path,
                    source,
                    "prediction_confidence_out_of_range",
                    f"confidence {conf} is outside [0, 1].",
                    line_no=idx,
                )
            )
    return issues


def _validate_normalized_bbox(
    *,
    image_path: Path,
    file_path: Path,
    source: str,
    line_no: int,
    xc: float,
    yc: float,
    w: float,
    h: float,
) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []
    if not (0.0 <= xc <= 1.0) or not (0.0 <= yc <= 1.0):
        issues.append(
            _issue(
                image_path,
                file_path,
                source,
                f"{source}_center_out_of_range",
                f"center ({xc}, {yc}) is outside [0, 1].",
                line_no=line_no,
            )
        )
    if w <= 0.0 or h <= 0.0 or w > 1.0 or h > 1.0:
        issues.append(
            _issue(
                image_path,
                file_path,
                source,
                f"{source}_size_out_of_range",
                f"width/height ({w}, {h}) must be within (0, 1].",
                line_no=line_no,
            )
        )
    left = xc - w / 2.0
    right = xc + w / 2.0
    top = yc - h / 2.0
    bottom = yc + h / 2.0
    if left < 0.0 or right > 1.0 or top < 0.0 or bottom > 1.0:
        issues.append(
            _issue(
                image_path,
                file_path,
                source,
                f"{source}_bbox_out_of_bounds",
                f"bbox extents ({left:.4f}, {top:.4f}, {right:.4f}, {bottom:.4f}) exceed [0, 1].",
                line_no=line_no,
            )
        )
    return issues


def _issue(
    image_path: Path,
    file_path: Path | str,
    source: str,
    issue_type: str,
    detail: str,
    *,
    line_no: int = 0,
) -> ValidationIssue:
    return ValidationIssue(
        image_path=str(image_path),
        file_path=str(file_path),
        source=source,
        issue_type=issue_type,
        detail=detail,
        line_no=int(line_no),
    )
