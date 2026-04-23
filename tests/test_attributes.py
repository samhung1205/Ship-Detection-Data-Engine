import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.attributes import (  # noqa: E402
    bbox_area_px,
    compute_size_tag,
    default_attributes_dict,
    normalize_attributes,
)


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


def test_default_attributes_include_hard_example_flags() -> None:
    attrs = default_attributes_dict()
    assert attrs["hard_sample"] == "false"
    assert attrs["occluded"] == "false"
    assert attrs["truncated"] == "false"
    assert attrs["blurred"] == "false"
    assert attrs["difficult_background"] == "false"
    assert attrs["low_contrast"] == "false"


def test_normalize_attributes_coerces_boolean_flags() -> None:
    attrs = normalize_attributes(
        {
            "crowded": "yes",
            "hard_sample": "on",
            "occluded": 1,
            "truncated": "n",
            "blurred": "TRUE",
            "difficult_background": True,
            "low_contrast": "0",
        }
    )
    assert attrs["crowded"] == "true"
    assert attrs["hard_sample"] == "true"
    assert attrs["occluded"] == "true"
    assert attrs["truncated"] == "false"
    assert attrs["blurred"] == "true"
    assert attrs["difficult_background"] == "true"
    assert attrs["low_contrast"] == "false"


if __name__ == "__main__":
    test_size_tag_thresholds()
    test_area()
    test_default_attributes_include_hard_example_flags()
    test_normalize_attributes_coerces_boolean_flags()
    print("OK")
