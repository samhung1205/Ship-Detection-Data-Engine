import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.attributes import bbox_area_px, compute_size_tag  # noqa: E402


def test_size_tag_thresholds() -> None:
    # area 31*31 < 32*32 -> small
    assert compute_size_tag(0, 0, 31, 31) == "small"
    # 32x32 = 1024, not < 1024 -> medium band
    assert compute_size_tag(0, 0, 32, 32) == "medium"
    # 96x96 = 9216, within medium upper bound
    assert compute_size_tag(0, 0, 96, 96) == "medium"
    assert compute_size_tag(0, 0, 97, 97) == "large"


def test_area() -> None:
    assert bbox_area_px(0, 0, 10, 10) == 100.0


if __name__ == "__main__":
    test_size_tag_thresholds()
    test_area()
    print("OK")
