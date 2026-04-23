import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sdde.model_inference import load_yolo_model, run_yolo_model_inference  # noqa: E402


class _FakeTensor:
    def __init__(self, data):
        self._data = data

    def cpu(self):
        return self

    def tolist(self):
        return self._data


class _FakeBoxes:
    def __init__(self):
        self.xyxy = _FakeTensor([[10, 20, 50, 80], [5, 6, 25, 36]])
        self.cls = _FakeTensor([1, 0])
        self.conf = _FakeTensor([0.9, 0.75])


class _FakeResult:
    def __init__(self):
        self.boxes = _FakeBoxes()


class _FakeModel:
    def __init__(self, path: str):
        self.path = path
        self.calls = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [_FakeResult()]


def test_load_yolo_model_uses_optional_backend_loader(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "ship.pt"
    model_path.write_bytes(b"fake-model")

    monkeypatch.setattr(
        "sdde.model_inference._load_ultralytics_yolo_class",
        lambda: _FakeModel,
    )

    handle = load_yolo_model(model_path)

    assert handle.model_path == str(model_path)
    assert handle.backend_name == "ultralytics"
    assert isinstance(handle.model, _FakeModel)


def test_run_yolo_model_inference_returns_prediction_records(tmp_path, monkeypatch) -> None:
    model_path = tmp_path / "ship.pt"
    model_path.write_bytes(b"fake-model")
    image_path = tmp_path / "scene.jpg"
    image_path.write_bytes(b"fake-image")

    monkeypatch.setattr(
        "sdde.model_inference._load_ultralytics_yolo_class",
        lambda: _FakeModel,
    )
    handle = load_yolo_model(model_path)

    preds = run_yolo_model_inference(
        handle,
        image_path=image_path,
        object_list=["naval", "merchant"],
        conf_threshold=0.15,
        iou_threshold=0.6,
        max_det=50,
    )

    assert len(preds) == 2
    assert preds[0].class_id == 1
    assert preds[0].class_name == "merchant"
    assert preds[0].x1 == 10
    assert preds[0].y2 == 80
    assert preds[0].confidence == 0.9
    assert handle.model.calls[0]["source"] == str(image_path)
    assert handle.model.calls[0]["conf"] == 0.15
    assert handle.model.calls[0]["iou"] == 0.6
    assert handle.model.calls[0]["max_det"] == 50
