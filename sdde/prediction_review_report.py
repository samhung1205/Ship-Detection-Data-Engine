"""
Batch reporting helpers for folder / project prediction review workflows.
"""
from __future__ import annotations

import csv
import io
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .dataset_scan import _list_image_paths
from .prediction_review import PredictionReviewState, prediction_review_status
from .prediction_review_store import load_prediction_review_session


@dataclass(frozen=True)
class PredictionReviewReportEntry:
    image_path: str
    prediction_path: str
    has_prediction: bool
    status: str
    original_count: int
    accepted_count: int
    rejected_count: int
    remaining_count: int


@dataclass(frozen=True)
class PredictionReviewReport:
    scope_path: str
    prediction_root: str
    image_paths: tuple[str, ...]
    entries: tuple[PredictionReviewReportEntry, ...]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def images_with_predictions(self) -> int:
        return sum(1 for entry in self.entries if entry.has_prediction)

    @property
    def reviewed_images(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "reviewed")

    @property
    def partial_images(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "partial")

    @property
    def pending_images(self) -> int:
        return sum(1 for entry in self.entries if entry.status == "pending")

    @property
    def no_prediction_images(self) -> int:
        return sum(1 for entry in self.entries if not entry.has_prediction)

    @property
    def total_original_predictions(self) -> int:
        return sum(entry.original_count for entry in self.entries)

    @property
    def total_accepted_predictions(self) -> int:
        return sum(entry.accepted_count for entry in self.entries)

    @property
    def total_rejected_predictions(self) -> int:
        return sum(entry.rejected_count for entry in self.entries)

    @property
    def total_remaining_predictions(self) -> int:
        return sum(entry.remaining_count for entry in self.entries)


def scan_prediction_review_report(
    scope_path: str | Path,
    *,
    prediction_root: str | Path,
    review_root: str | Path,
    recursive: bool = False,
    image_root: str | Path | None = None,
    current_states: Mapping[str, PredictionReviewState] | None = None,
) -> PredictionReviewReport:
    root = Path(scope_path)
    image_paths = tuple(_list_image_paths(root, recursive=recursive))
    prediction_root_path = Path(prediction_root).resolve()
    current_state_map = {str(Path(k).resolve()): v for k, v in (current_states or {}).items()}
    folder_state_cache: dict[str, dict[str, PredictionReviewState]] = {}
    entries: list[PredictionReviewReportEntry] = []

    for image_path_str in image_paths:
        image_path = Path(image_path_str).resolve()
        key = str(image_path)
        pred_path = _prediction_path_for_image(
            image_path,
            prediction_root=prediction_root_path,
            image_root=image_root,
        )
        has_prediction = pred_path.is_file()
        if key in current_state_map:
            state = current_state_map[key]
        else:
            folder_key = str(image_path.parent.resolve())
            if folder_key not in folder_state_cache:
                folder_state_cache[folder_key] = load_prediction_review_session(
                    image_folder=image_path.parent,
                    prediction_folder=prediction_root_path,
                    review_root=review_root,
                ) or {}
            state = folder_state_cache[folder_key].get(key)

        if state is not None:
            status = prediction_review_status(state)
            original_count = state.original_count
            accepted_count = state.accepted_count
            rejected_count = state.rejected_count
            remaining_count = state.remaining_count
        elif has_prediction:
            original_count = _count_prediction_rows(pred_path)
            accepted_count = 0
            rejected_count = 0
            remaining_count = original_count
            status = "pending"
        else:
            original_count = 0
            accepted_count = 0
            rejected_count = 0
            remaining_count = 0
            status = "no_predictions"

        entries.append(
            PredictionReviewReportEntry(
                image_path=str(image_path),
                prediction_path=str(pred_path),
                has_prediction=has_prediction,
                status=status,
                original_count=original_count,
                accepted_count=accepted_count,
                rejected_count=rejected_count,
                remaining_count=remaining_count,
            )
        )

    return PredictionReviewReport(
        scope_path=str(root.resolve()),
        prediction_root=str(prediction_root_path),
        image_paths=image_paths,
        entries=tuple(entries),
    )


def build_prediction_review_report_summary(report: PredictionReviewReport) -> dict[str, Any]:
    return {
        "scope_path": report.scope_path,
        "prediction_root": report.prediction_root,
        "total_images": report.total_images,
        "images_with_predictions": report.images_with_predictions,
        "reviewed_images": report.reviewed_images,
        "partial_images": report.partial_images,
        "pending_images": report.pending_images,
        "no_prediction_images": report.no_prediction_images,
        "total_original_predictions": report.total_original_predictions,
        "total_accepted_predictions": report.total_accepted_predictions,
        "total_rejected_predictions": report.total_rejected_predictions,
        "total_remaining_predictions": report.total_remaining_predictions,
    }


def export_prediction_review_report_csv(report: PredictionReviewReport) -> str:
    out = io.StringIO()
    writer = csv.writer(out)
    writer.writerow(
        [
            "image_path",
            "prediction_path",
            "has_prediction",
            "status",
            "original_count",
            "accepted_count",
            "rejected_count",
            "remaining_count",
        ]
    )
    for entry in report.entries:
        writer.writerow(
            [
                entry.image_path,
                entry.prediction_path,
                str(entry.has_prediction).lower(),
                entry.status,
                entry.original_count,
                entry.accepted_count,
                entry.rejected_count,
                entry.remaining_count,
            ]
        )
    return out.getvalue()


def export_prediction_review_report_json(report: PredictionReviewReport) -> str:
    payload = build_prediction_review_report_summary(report)
    payload["entries"] = [
        {
            "image_path": entry.image_path,
            "prediction_path": entry.prediction_path,
            "has_prediction": entry.has_prediction,
            "status": entry.status,
            "original_count": entry.original_count,
            "accepted_count": entry.accepted_count,
            "rejected_count": entry.rejected_count,
            "remaining_count": entry.remaining_count,
        }
        for entry in report.entries
    ]
    return json.dumps(payload, ensure_ascii=False, indent=2)


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


def _count_prediction_rows(prediction_path: Path) -> int:
    try:
        lines = prediction_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return 0
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))
