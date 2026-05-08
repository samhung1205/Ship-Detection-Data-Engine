"""Tests for session-level prediction review state helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.prediction import PredictionRecord
from sdde.prediction_review import (
    clone_predictions,
    initial_prediction_review_state,
    prediction_review_status,
    prediction_review_summary,
    update_prediction_review_state,
)


def _pred(pred_id: str) -> PredictionRecord:
    return PredictionRecord(
        pred_id=pred_id,
        class_id=0,
        class_name="naval",
        x1=0,
        y1=0,
        x2=10,
        y2=10,
        confidence=0.9,
    )


def test_initial_prediction_review_state_is_pending() -> None:
    state = initial_prediction_review_state([_pred("p1"), _pred("p2")])

    assert state.original_count == 2
    assert state.remaining_count == 2
    assert prediction_review_status(state) == "pending"


def test_update_prediction_review_state_tracks_partial_and_reviewed() -> None:
    state = initial_prediction_review_state([_pred("p1"), _pred("p2")])
    partial = update_prediction_review_state(
        state,
        accepted_delta=1,
        remaining_predictions=[_pred("p2")],
    )
    reviewed = update_prediction_review_state(
        partial,
        rejected_delta=1,
        remaining_predictions=[],
    )

    assert prediction_review_status(partial) == "partial"
    assert prediction_review_status(reviewed) == "reviewed"
    assert "accepted 1" in prediction_review_summary(reviewed)
    assert "rejected 1" in prediction_review_summary(reviewed)


def test_clone_predictions_returns_detached_copies() -> None:
    preds = [_pred("p1")]

    cloned = clone_predictions(preds)
    cloned[0].class_name = "merchant"

    assert preds[0].class_name == "naval"
