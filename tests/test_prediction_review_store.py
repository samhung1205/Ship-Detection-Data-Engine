"""Tests for persisted folder-level prediction review sessions."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.prediction import PredictionRecord
from sdde.prediction_review import initial_prediction_review_state, update_prediction_review_state
from sdde.prediction_review_store import (
    has_prediction_review_session,
    load_prediction_review_session,
    remove_prediction_review_session,
    review_state_path,
    save_prediction_review_session,
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


def test_prediction_review_session_roundtrip(tmp_path: Path) -> None:
    states = {
        str((tmp_path / "images" / "a.jpg").resolve()): initial_prediction_review_state([_pred("p1"), _pred("p2")]),
        str((tmp_path / "images" / "b.jpg").resolve()): update_prediction_review_state(
            initial_prediction_review_state([_pred("p3")]),
            rejected_delta=1,
            remaining_predictions=[],
        ),
    }

    path = save_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
        states=states,
    )

    loaded = load_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
    )

    assert path == review_state_path(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
    )
    assert loaded is not None
    assert loaded[str((tmp_path / "images" / "a.jpg").resolve())].remaining_count == 2
    assert loaded[str((tmp_path / "images" / "b.jpg").resolve())].rejected_count == 1
    assert has_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
    ) is True


def test_remove_prediction_review_session_deletes_saved_file(tmp_path: Path) -> None:
    save_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
        states={str((tmp_path / "images" / "a.jpg").resolve()): initial_prediction_review_state([_pred("p1")])},
    )

    remove_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
    )

    assert has_prediction_review_session(
        image_folder=tmp_path / "images",
        prediction_folder=tmp_path / "preds",
        review_root=tmp_path / "project",
    ) is False
