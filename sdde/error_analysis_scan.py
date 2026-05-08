"""
Folder-level error analysis scanning helpers.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

from .dataset_scan import ImageAnnotationBundle, _list_image_paths, load_image_annotation_bundle, read_image_size
from .error_analysis import ErrorCase, match_gt_pred
from .prediction import PredictionRecord, filter_predictions_by_confidence, parse_predictions_yolo_txt


@dataclass(frozen=True)
class FolderErrorAnalysisResult:
    folder_path: str
    prediction_root: str
    image_paths: tuple[str, ...]
    analyzed_image_paths: tuple[str, ...]
    labeled_image_paths: tuple[str, ...]
    prediction_image_paths: tuple[str, ...]
    cases: tuple[ErrorCase, ...]

    @property
    def total_images(self) -> int:
        return len(self.image_paths)

    @property
    def analyzed_images(self) -> int:
        return len(self.analyzed_image_paths)

    @property
    def labeled_images(self) -> int:
        return len(self.labeled_image_paths)

    @property
    def prediction_images(self) -> int:
        return len(self.prediction_image_paths)


def scan_folder_error_cases(
    folder: str | Path,
    *,
    object_list: Sequence[str],
    prediction_root: str | Path,
    recursive: bool = False,
    image_root: str | Path | None = None,
    label_root: str | Path | None = None,
    current_image_path: str | Path | None = None,
    current_image_gt_bundle: ImageAnnotationBundle | None = None,
    current_image_predictions: Sequence[PredictionRecord] | None = None,
    min_confidence: float = 0.0,
) -> FolderErrorAnalysisResult:
    folder_path = Path(folder)
    prediction_root_path = Path(prediction_root)
    image_paths = tuple(_list_image_paths(folder_path, recursive=recursive))
    current_image_resolved = _resolve_path_or_none(current_image_path)
    analyzed_image_paths: list[str] = []
    labeled_image_paths: list[str] = []
    prediction_image_paths: list[str] = []
    cases: list[ErrorCase] = []

    for image_path_str in image_paths:
        image_path = Path(image_path_str).resolve()
        if current_image_resolved is not None and image_path == current_image_resolved and current_image_gt_bundle is not None:
            gt_bundle = current_image_gt_bundle
        else:
            gt_bundle = load_image_annotation_bundle(
                image_path,
                object_list=object_list,
                image_root=image_root,
                label_root=label_root,
            )
        if gt_bundle.has_label:
            labeled_image_paths.append(str(image_path))

        if current_image_resolved is not None and image_path == current_image_resolved and current_image_predictions is not None:
            predictions = filter_predictions_by_confidence(
                current_image_predictions,
                min_confidence=min_confidence,
            )
        else:
            predictions = _load_prediction_sidecar(
                image_path,
                prediction_root=prediction_root_path,
                image_root=image_root,
                object_list=object_list,
                min_confidence=min_confidence,
            )
        if predictions:
            prediction_image_paths.append(str(image_path))

        if not gt_bundle.has_label and not predictions:
            continue

        analyzed_image_paths.append(str(image_path))
        cases.extend(
            match_gt_pred(
                gt_bundle.gt_boxes,
                predictions,
                gt_attributes=gt_bundle.gt_attributes,
                image_id=str(image_path),
            )
        )

    return FolderErrorAnalysisResult(
        folder_path=str(folder_path),
        prediction_root=str(prediction_root_path),
        image_paths=image_paths,
        analyzed_image_paths=tuple(analyzed_image_paths),
        labeled_image_paths=tuple(labeled_image_paths),
        prediction_image_paths=tuple(prediction_image_paths),
        cases=tuple(cases),
    )


def _load_prediction_sidecar(
    image_path: Path,
    *,
    prediction_root: Path,
    image_root: str | Path | None,
    object_list: Sequence[str],
    min_confidence: float,
) -> list[PredictionRecord]:
    size = read_image_size(image_path)
    if size is None:
        return []
    image_w, image_h = size
    pred_path = next(
        (
            candidate
            for candidate in _prediction_sidecar_paths(
                image_path,
                prediction_root=prediction_root,
                image_root=image_root,
            )
            if candidate.is_file()
        ),
        None,
    )
    if pred_path is None:
        return []
    try:
        body = pred_path.read_text(encoding="utf-8")
        predictions = parse_predictions_yolo_txt(
            body,
            object_list=object_list,
            image_w=image_w,
            image_h=image_h,
        )
    except (OSError, ValueError, IndexError):
        return []
    return filter_predictions_by_confidence(predictions, min_confidence=min_confidence)


def _resolve_path_or_none(path: str | Path | None) -> Path | None:
    if not path:
        return None
    return Path(path).resolve()


def _prediction_sidecar_path(
    image_path: Path,
    *,
    prediction_root: Path,
    image_root: str | Path | None,
) -> Path:
    return _prediction_sidecar_paths(
        image_path,
        prediction_root=prediction_root,
        image_root=image_root,
    )[0]


def _prediction_sidecar_paths(
    image_path: Path,
    *,
    prediction_root: Path,
    image_root: str | Path | None,
) -> list[Path]:
    candidates: list[Path] = []
    if image_root:
        try:
            rel = image_path.resolve().relative_to(Path(image_root).resolve())
        except ValueError:
            rel = None
        if rel is not None:
            candidates.append(prediction_root / rel.parent / f"{image_path.stem}.txt")
    candidates.append(prediction_root / f"{image_path.stem}.txt")

    out: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate)
        if key in seen:
            continue
        seen.add(key)
        out.append(candidate)
    return out
