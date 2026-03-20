import sys
from pathlib import Path

# Allow running this file directly: `python tests/test_models.py`
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.models import ClassMapping, HBBBoxPx


def test_class_mapping_default() -> None:
    m = ClassMapping.default_ship_mapping()
    assert m.nc == 4
    assert m.id_to_name(0) == "Naval"
    assert m.name_to_id("Dock") == 2


def test_hbb_to_yolo_norm() -> None:
    # 960x960 common in prompt.md
    image_w = 960
    image_h = 960
    # bbox covering left half vertically (y: 0..480), and full width (x: 0..960)
    bbox = HBBBoxPx(x1=0, y1=0, x2=960, y2=480)
    norm = bbox.to_yolo_norm(image_w=image_w, image_h=image_h)

    assert abs(norm.x_center - 0.5) < 1e-9
    assert abs(norm.y_center - 0.25) < 1e-9
    assert abs(norm.width - 1.0) < 1e-9
    assert abs(norm.height - 0.5) < 1e-9


if __name__ == "__main__":
    test_class_mapping_default()
    test_hbb_to_yolo_norm()
    print("OK")

