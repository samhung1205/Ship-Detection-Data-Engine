"""
Persistence helpers for folder-level prediction review sessions.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .prediction import PredictionRecord
from .prediction_review import PredictionReviewState


REVIEW_STATE_DIR = ".review_state"
REVIEW_STATE_SUFFIX = ".prediction_review.json"


def review_state_path(
    *,
    image_folder: str | Path,
    prediction_folder: str | Path,
    review_root: str | Path,
) -> Path:
    img = Path(image_folder).resolve()
    pred = Path(prediction_folder).resolve()
    root = Path(review_root).resolve()
    state_dir = root / REVIEW_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{img}|{pred}".encode("utf-8")).hexdigest()[:16]
    return state_dir / f"{key}{REVIEW_STATE_SUFFIX}"


def save_prediction_review_session(
    *,
    image_folder: str | Path,
    prediction_folder: str | Path,
    review_root: str | Path,
    states: Mapping[str, PredictionReviewState],
) -> Path:
    payload = {
        "image_folder": str(Path(image_folder).resolve()),
        "prediction_folder": str(Path(prediction_folder).resolve()),
        "states": {
            str(Path(image_path).resolve()): _state_to_dict(state)
            for image_path, state in states.items()
        },
    }
    path = review_state_path(
        image_folder=image_folder,
        prediction_folder=prediction_folder,
        review_root=review_root,
    )
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def load_prediction_review_session(
    *,
    image_folder: str | Path,
    prediction_folder: str | Path,
    review_root: str | Path,
) -> dict[str, PredictionReviewState] | None:
    path = review_state_path(
        image_folder=image_folder,
        prediction_folder=prediction_folder,
        review_root=review_root,
    )
    if not path.exists():
        return None
    try:
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if not isinstance(payload, Mapping):
        return None
    raw_states = payload.get("states")
    if not isinstance(raw_states, Mapping):
        return None
    states: dict[str, PredictionReviewState] = {}
    for image_path, raw_state in raw_states.items():
        if not isinstance(image_path, str) or not isinstance(raw_state, Mapping):
            continue
        states[str(Path(image_path).resolve())] = _state_from_dict(raw_state)
    return states


def remove_prediction_review_session(
    *,
    image_folder: str | Path,
    prediction_folder: str | Path,
    review_root: str | Path,
) -> None:
    path = review_state_path(
        image_folder=image_folder,
        prediction_folder=prediction_folder,
        review_root=review_root,
    )
    if path.exists():
        path.unlink(missing_ok=True)


def has_prediction_review_session(
    *,
    image_folder: str | Path,
    prediction_folder: str | Path,
    review_root: str | Path,
) -> bool:
    return review_state_path(
        image_folder=image_folder,
        prediction_folder=prediction_folder,
        review_root=review_root,
    ).exists()


def _prediction_to_dict(pred: PredictionRecord) -> dict[str, Any]:
    return {
        "pred_id": pred.pred_id,
        "class_id": int(pred.class_id),
        "class_name": pred.class_name,
        "x1": float(pred.x1),
        "y1": float(pred.y1),
        "x2": float(pred.x2),
        "y2": float(pred.y2),
        "confidence": float(pred.confidence),
        "pred_status": pred.pred_status,
    }


def _prediction_from_dict(raw: Mapping[str, Any]) -> PredictionRecord:
    return PredictionRecord(
        pred_id=str(raw.get("pred_id", "")),
        class_id=int(raw.get("class_id", 0)),
        class_name=str(raw.get("class_name", "")),
        x1=float(raw.get("x1", 0.0)),
        y1=float(raw.get("y1", 0.0)),
        x2=float(raw.get("x2", 0.0)),
        y2=float(raw.get("y2", 0.0)),
        confidence=float(raw.get("confidence", 0.0)),
        pred_status=str(raw.get("pred_status", "predicted")),
    )


def _state_to_dict(state: PredictionReviewState) -> dict[str, Any]:
    return {
        "original_count": int(state.original_count),
        "accepted_count": int(state.accepted_count),
        "rejected_count": int(state.rejected_count),
        "remaining_predictions": [_prediction_to_dict(pred) for pred in state.remaining_predictions],
    }


def _state_from_dict(raw: Mapping[str, Any]) -> PredictionReviewState:
    remaining_raw = raw.get("remaining_predictions") or []
    remaining = [
        _prediction_from_dict(item)
        for item in remaining_raw
        if isinstance(item, Mapping)
    ]
    return PredictionReviewState(
        original_count=int(raw.get("original_count", len(remaining))),
        accepted_count=int(raw.get("accepted_count", 0)),
        rejected_count=int(raw.get("rejected_count", 0)),
        remaining_predictions=tuple(remaining),
    )
