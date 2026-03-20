import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.import_export import export_bbox_txt, export_yolo_hbb_txt, parse_yolo_hbb_txt
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
    assert out_name.strip() == "Dock 10 20 30 40"


if __name__ == "__main__":
    test_yolo_import_export_roundtrip()
    test_bbox_export_class_mode()
    print("OK")

