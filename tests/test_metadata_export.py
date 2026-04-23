import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.metadata_export import (  # noqa: E402
    build_annotation_records,
    export_annotations_csv,
    export_annotations_json,
)


def test_build_and_export() -> None:
    real = [["naval", 0.0, 0.0, 10.0, 10.0]]
    attrs = [
        {
            "size_tag": "small",
            "crowded": "false",
            "difficulty_tag": "hard",
            "hard_sample": "true",
            "occluded": "true",
            "truncated": "false",
            "blurred": "true",
            "difficult_background": "true",
            "low_contrast": "false",
            "scene_tag": "unknown",
        }
    ]
    recs = build_annotation_records(
        image_path="/tmp/a.jpg",
        image_width=960,
        image_height=960,
        real_data=real,
        box_attributes=attrs,
        object_list=["naval", "merchant"],
        class_id_to_super={0: "vessel"},
    )
    assert recs[0]["class_id"] == 0
    assert recs[0]["super_category"] == "vessel"
    assert recs[0]["difficulty_tag"] == "hard"
    assert recs[0]["hard_sample"] == "true"
    assert recs[0]["occluded"] == "true"
    assert recs[0]["truncated"] == "false"
    assert recs[0]["blurred"] == "true"
    assert recs[0]["difficult_background"] == "true"
    assert recs[0]["low_contrast"] == "false"
    assert "naval" in export_annotations_json(recs)
    csv_text = export_annotations_csv(recs)
    assert "naval" in csv_text
    assert "hard_sample" in csv_text
    assert "difficult_background" in csv_text


def test_build_annotation_records_computes_size_tag_when_missing() -> None:
    recs = build_annotation_records(
        image_path="/tmp/a.jpg",
        image_width=960,
        image_height=960,
        real_data=[["naval", 0.0, 0.0, 10.0, 10.0]],
        box_attributes=[{}],
        object_list=["naval"],
    )

    assert recs[0]["size_tag"] == "small"


if __name__ == "__main__":
    test_build_and_export()
    print("OK")
