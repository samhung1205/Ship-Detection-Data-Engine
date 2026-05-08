"""
Session-level prediction review state helpers.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Sequence

from .prediction import PredictionRecord


@dataclass(frozen=True)
class PredictionReviewState:
    original_count: int
    accepted_count: int = 0
    rejected_count: int = 0
    remaining_predictions: tuple[PredictionRecord, ...] = ()

    @property
    def remaining_count(self) -> int:
        return len(self.remaining_predictions)


def clone_predictions(predictions: Sequence[PredictionRecord]) -> list[PredictionRecord]:
    return [copy.deepcopy(pred) for pred in predictions]


def initial_prediction_review_state(
    predictions: Sequence[PredictionRecord],
) -> PredictionReviewState:
    copied = tuple(clone_predictions(predictions))
    return PredictionReviewState(
        original_count=len(copied),
        remaining_predictions=copied,
    )


def update_prediction_review_state(
    state: PredictionReviewState,
    *,
    accepted_delta: int = 0,
    rejected_delta: int = 0,
    remaining_predictions: Sequence[PredictionRecord] | None = None,
) -> PredictionReviewState:
    copied = (
        tuple(clone_predictions(remaining_predictions))
        if remaining_predictions is not None
        else state.remaining_predictions
    )
    return PredictionReviewState(
        original_count=state.original_count,
        accepted_count=max(0, state.accepted_count + int(accepted_delta)),
        rejected_count=max(0, state.rejected_count + int(rejected_delta)),
        remaining_predictions=copied,
    )


def prediction_review_status(state: PredictionReviewState) -> str:
    if state.original_count <= 0:
        return "pending"
    if state.remaining_count == 0 and (state.accepted_count + state.rejected_count) >= state.original_count:
        return "reviewed"
    if state.accepted_count > 0 or state.rejected_count > 0:
        return "partial"
    return "pending"


def prediction_review_summary(state: PredictionReviewState) -> str:
    return (
        f"{prediction_review_status(state)}"
        f" | accepted {state.accepted_count}"
        f" | rejected {state.rejected_count}"
        f" | remaining {state.remaining_count}"
    )
