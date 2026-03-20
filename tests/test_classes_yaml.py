import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.classes_yaml import load_classes_yaml, save_classes_yaml  # noqa: E402
from sdde.class_catalog import default_ship_catalog  # noqa: E402


def test_roundtrip_default() -> None:
    cat = default_ship_catalog()
    text = save_classes_yaml(cat)
    cat2 = load_classes_yaml(text)
    assert cat2.signature() == cat.signature()
    assert cat2.project_name == cat.project_name


def test_load_repo_classes_yaml() -> None:
    p = ROOT / "classes.yaml"
    text = p.read_text(encoding="utf-8")
    cat = load_classes_yaml(text)
    assert len(cat.classes) == 4
    assert cat.names_ordered()[0] == "naval"


if __name__ == "__main__":
    test_roundtrip_default()
    test_load_repo_classes_yaml()
    print("OK")
