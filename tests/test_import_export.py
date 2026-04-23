import json
import sys
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.import_export import (
    export_bbox_txt,
    export_coco_bbox_json,
    export_pascal_voc_xml,
    export_yolo_hbb_txt,
    parse_annotation_metadata_json,
    parse_coco_bbox_json,
    parse_yolo_hbb_txt,
)
from sdde.models import ClassMapping, HBBAnnotation, HBBBoxPx


def test_yolo_import_export_roundtrip() -> None:
    mapping = ClassMapping.default_ship_mapping()
    image_w = 960
    image_h = 960

    # One bbox: x1=0,y1=0,x2=960,y2=480
    ann = HBBAnnotation(
        class_id=0,
        bbox_px=HBBBoxPx(x1=0, y1=0, x2=960, y2=480),
    )

    yolo_txt = export_yolo_hbb_txt([ann], class_mapping=mapping, image_w=image_w, image_h=image_h)
    imported = parse_yolo_hbb_txt(
        yolo_txt,
        class_mapping=mapping,
        image_w=image_w,
        image_h=image_h,
    )

    assert len(imported) == 1
    assert imported[0].class_id == 0
    # Compare with tolerance; conversion uses floats.
    assert abs(imported[0].bbox_px.x1 - 0) < 1e-4
    assert abs(imported[0].bbox_px.y1 - 0) < 1e-4
    assert abs(imported[0].bbox_px.x2 - 960) < 1e-4
    assert abs(imported[0].bbox_px.y2 - 480) < 1e-4


def test_bbox_export_class_mode() -> None:
    mapping = ClassMapping.default_ship_mapping()

    ann = HBBAnnotation(
        class_id=2,
        bbox_px=HBBBoxPx(x1=10, y1=20, x2=30, y2=40),
    )

    out_id = export_bbox_txt([ann], class_mapping=mapping, cls_mode="class_id")
    assert out_id.strip() == "2 10 20 30 40"

    out_name = export_bbox_txt([ann], class_mapping=mapping, cls_mode="class_name")
    assert out_name.strip() == "dock 10 20 30 40"


def test_bbox_export_coords_first_order() -> None:
    mapping = ClassMapping.default_ship_mapping()

    ann = HBBAnnotation(
        class_id=2,
        bbox_px=HBBBoxPx(x1=10, y1=20, x2=30, y2=40),
    )

    out = export_bbox_txt(
        [ann],
        class_mapping=mapping,
        cls_mode="class_id",
        order="coords_first",
    )
    assert out.strip() == "10 20 30 40 2"


def test_coco_bbox_export_includes_single_image_annotations_and_categories() -> None:
    mapping = ClassMapping.default_ship_mapping()
    ann = HBBAnnotation(
        class_id=1,
        bbox_px=HBBBoxPx(x1=10, y1=20, x2=50, y2=80),
        attributes={"crowded": "true"},
    )

    text = export_coco_bbox_json(
        [ann],
        class_mapping=mapping,
        image_w=640,
        image_h=480,
        image_path="/tmp/sample.jpg",
    )
    data = json.loads(text)

    assert data["images"] == [{"id": 1, "file_name": "sample.jpg", "width": 640, "height": 480}]
    assert data["annotations"][0]["category_id"] == 1
    assert data["annotations"][0]["bbox"] == [10.0, 20.0, 40.0, 60.0]
    assert data["annotations"][0]["area"] == 2400.0
    assert data["annotations"][0]["iscrowd"] == 1
    assert data["categories"][1]["name"] == "merchant"


def test_coco_bbox_import_matches_current_image_filename_and_iscrowd() -> None:
    mapping = ClassMapping.default_ship_mapping()
    payload = {
        "images": [
            {"id": 1, "file_name": "other.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "sample.jpg", "width": 640, "height": 480},
        ],
        "categories": [
            {"id": 10, "name": "merchant"},
            {"id": 11, "name": "dock"},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 10, "bbox": [1, 2, 3, 4], "iscrowd": 0},
            {"id": 2, "image_id": 2, "category_id": 10, "bbox": [10, 20, 40, 60], "iscrowd": 1},
        ],
    }

    imported = parse_coco_bbox_json(
        json.dumps(payload),
        class_mapping=mapping,
        image_w=640,
        image_h=480,
        image_path="/tmp/sample.jpg",
    )

    assert len(imported) == 1
    assert imported[0].class_id == 1
    assert imported[0].bbox_px == HBBBoxPx(x1=10, y1=20, x2=50, y2=80)
    assert imported[0].attributes["crowded"] == "true"


def test_coco_bbox_import_accepts_single_image_project_export() -> None:
    mapping = ClassMapping.default_ship_mapping()
    text = export_coco_bbox_json(
        [HBBAnnotation(class_id=0, bbox_px=HBBBoxPx(x1=5, y1=6, x2=25, y2=36))],
        class_mapping=mapping,
        image_w=320,
        image_h=240,
        image_path="/tmp/sample.jpg",
    )

    imported = parse_coco_bbox_json(
        text,
        class_mapping=mapping,
        image_w=320,
        image_h=240,
        image_path="/tmp/sample.jpg",
    )

    assert len(imported) == 1
    assert imported[0].class_id == 0
    assert imported[0].bbox_px == HBBBoxPx(x1=5, y1=6, x2=25, y2=36)


def test_annotation_metadata_json_import_restores_attributes() -> None:
    mapping = ClassMapping.default_ship_mapping()
    payload = [
        {
            "image_path": "/tmp/sample.jpg",
            "image_width": 640,
            "image_height": 480,
            "class_name": "merchant",
            "class_id": 1,
            "x1": 10,
            "y1": 20,
            "x2": 50,
            "y2": 80,
            "size_tag": "small",
            "crowded": "true",
            "difficulty_tag": "hard",
            "hard_sample": "true",
            "occluded": "true",
            "truncated": "false",
            "blurred": "true",
            "difficult_background": "true",
            "low_contrast": "false",
            "scene_tag": "near_shore",
        }
    ]

    imported = parse_annotation_metadata_json(
        json.dumps(payload),
        class_mapping=mapping,
        image_w=640,
        image_h=480,
        image_path="/tmp/sample.jpg",
    )

    assert len(imported) == 1
    assert imported[0].class_id == 1
    assert imported[0].bbox_px == HBBBoxPx(x1=10, y1=20, x2=50, y2=80)
    assert imported[0].attributes["crowded"] == "true"
    assert imported[0].attributes["difficulty_tag"] == "hard"
    assert imported[0].attributes["hard_sample"] == "true"
    assert imported[0].attributes["scene_tag"] == "near_shore"


def test_pascal_voc_export_includes_filename_and_objects() -> None:
    mapping = ClassMapping.default_ship_mapping()
    ann = HBBAnnotation(
        class_id=0,
        bbox_px=HBBBoxPx(x1=10, y1=20, x2=30, y2=40),
        attributes={"truncated": "true", "hard_sample": "true"},
    )

    text = export_pascal_voc_xml(
        [ann],
        class_mapping=mapping,
        image_w=320,
        image_h=240,
        image_path="/tmp/folder/sample.png",
    )
    root = ET.fromstring(text)

    assert root.findtext("folder") == "folder"
    assert root.findtext("filename") == "sample.png"
    assert root.findtext("size/width") == "320"
    assert root.findtext("size/height") == "240"
    assert root.findtext("object/name") == "naval"
    assert root.findtext("object/truncated") == "1"
    assert root.findtext("object/difficult") == "1"
    assert root.findtext("object/bndbox/xmin") == "10"
    assert root.findtext("object/bndbox/ymax") == "40"


if __name__ == "__main__":
    test_yolo_import_export_roundtrip()
    test_bbox_export_class_mode()
    test_bbox_export_coords_first_order()
    test_coco_bbox_export_includes_single_image_annotations_and_categories()
    test_coco_bbox_import_matches_current_image_filename_and_iscrowd()
    test_coco_bbox_import_accepts_single_image_project_export()
    test_annotation_metadata_json_import_restores_attributes()
    test_pascal_voc_export_includes_filename_and_objects()
    print("OK")
