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
    attrs = [{"size_tag": "small", "crowded": "false", "difficulty_tag": "normal", "scene_tag": "unknown"}]
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
    assert "naval" in export_annotations_json(recs)
    assert "naval" in export_annotations_csv(recs)


if __name__ == "__main__":
    test_build_and_export()
    print("OK")
