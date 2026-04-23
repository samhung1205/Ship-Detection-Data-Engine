"""Tests for folder-level annotation scanning helpers."""

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.dataset_scan import scan_folder_annotation_records


def _write_image(path: Path, *, width: int = 40, height: int = 20) -> None:
    img = np.zeros((height, width, 3), dtype=np.uint8)
    assert cv2.imwrite(str(path), img)


def test_scan_folder_annotation_records_reads_yolo_sidecars(tmp_path: Path) -> None:
    img_a = tmp_path / "a.jpg"
    img_b = tmp_path / "b.png"
    _write_image(img_a)
    _write_image(img_b)
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    result = scan_folder_annotation_records(
        tmp_path,
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel", 1: "vessel"},
    )

    assert result.total_images == 2
    assert result.labeled_images == 1
    assert len(result.records) == 1
    assert result.records[0]["class_name"] == "naval"
    assert result.records[0]["image_path"] == str(img_a)


def test_scan_folder_annotation_records_uses_current_image_override(tmp_path: Path) -> None:
    img_a = tmp_path / "a.jpg"
    _write_image(img_a)
    (tmp_path / "a.txt").write_text("0 0.5 0.5 0.5 0.5\n", encoding="utf-8")

    result = scan_folder_annotation_records(
        tmp_path,
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel", 1: "vessel"},
        current_image_path=img_a,
        current_image_records=[
            {
                "image_path": str(img_a),
                "image_width": 40,
                "image_height": 20,
                "class_name": "merchant",
                "class_id": 1,
                "super_category": "vessel",
                "x1": 1.0,
                "y1": 2.0,
                "x2": 5.0,
                "y2": 6.0,
                "size_tag": "small",
                "crowded": "false",
                "difficulty_tag": "normal",
                "hard_sample": "false",
                "occluded": "false",
                "truncated": "false",
                "blurred": "false",
                "difficult_background": "false",
                "low_contrast": "false",
                "scene_tag": "unknown",
                "annotation_source": "gt",
            }
        ],
    )

    assert result.total_images == 1
    assert result.labeled_images == 1
    assert len(result.records) == 1
    assert result.records[0]["class_name"] == "merchant"


def test_scan_folder_annotation_records_maps_image_root_to_label_root(tmp_path: Path) -> None:
    image_root = tmp_path / "images"
    label_root = tmp_path / "labels"
    folder = image_root / "train"
    folder.mkdir(parents=True)
    (label_root / "train").mkdir(parents=True)

    img = folder / "ship.jpg"
    _write_image(img)
    (label_root / "train" / "ship.txt").write_text("1 0.5 0.5 0.25 0.5\n", encoding="utf-8")

    result = scan_folder_annotation_records(
        folder,
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel", 1: "vessel"},
        image_root=image_root,
        label_root=label_root,
    )

    assert result.total_images == 1
    assert result.labeled_images == 1
    assert len(result.records) == 1
    assert result.records[0]["class_name"] == "merchant"
