"""Tests for folder-sidecar prediction helpers."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.prediction_scan import (
    has_prediction_sidecar,
    load_prediction_sidecar,
    prediction_sidecar_path,
)


def test_prediction_sidecar_path_uses_image_stem(tmp_path: Path) -> None:
    image = tmp_path / "folder" / "ship.jpg"
    pred_root = tmp_path / "preds"

    assert prediction_sidecar_path(image, prediction_root=pred_root) == pred_root / "ship.txt"


def test_has_and_load_prediction_sidecar(tmp_path: Path) -> None:
    image = tmp_path / "ship.jpg"
    pred_root = tmp_path / "preds"
    pred_root.mkdir()
    (pred_root / "ship.txt").write_text("0 0.5 0.5 0.5 0.5 0.9\n", encoding="utf-8")

    assert has_prediction_sidecar(image, prediction_root=pred_root) is True
    preds = load_prediction_sidecar(
        image,
        prediction_root=pred_root,
        object_list=["naval", "merchant"],
        image_w=100,
        image_h=100,
    )

    assert len(preds) == 1
    assert preds[0].class_name == "naval"
    assert preds[0].confidence == 0.9
