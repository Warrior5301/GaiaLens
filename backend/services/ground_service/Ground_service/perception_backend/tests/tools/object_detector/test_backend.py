"""Unit tests for DetectorBackend abstractions, MockBackend, and RemoteSensingYOLOBackend."""

from __future__ import annotations

from PIL import Image
from app.models.loaders import ModelLoader
from app.tools.object_detector.backend import MockDetectorBackend, RemoteSensingYOLOBackend
from sac_shared_contracts.common import BBoxPx, Confidence, Detection


def test_mock_detector_backend_lifecycle():
    backend = MockDetectorBackend(model_id="test_mock")
    assert not backend.is_loaded()

    backend.load()
    assert backend.is_loaded()

    img = Image.new("RGB", (500, 500), color=(200, 200, 200))
    dets = backend.predict(img, target_classes=["airplane"], confidence_threshold=0.3)
    assert len(dets) > 0
    assert all(d.class_name == "airplane" for d in dets)

    backend.unload()
    assert not backend.is_loaded()


def test_mock_detector_backend_preset():
    backend = MockDetectorBackend()
    preset = [
        Detection(
            object_id="custom_1",
            class_name="ship",
            confidence=Confidence(score=0.99, source="test"),
            bbox_px=BBoxPx(x_min=10, y_min=10, x_max=50, y_max=50),
        )
    ]
    backend.set_preset_detections(preset)

    img = Image.new("RGB", (200, 200))
    dets = backend.predict(img, target_classes=["ship"], confidence_threshold=0.5)
    assert len(dets) == 1
    assert dets[0].object_id == "custom_1"


def test_remote_sensing_yolo_backend():
    backend = RemoteSensingYOLOBackend(model_id="yolo_vhr_v1", device="cpu")
    backend.load()
    assert backend.is_loaded()

    from pathlib import Path
    img_path = Path("examples/real_satellite/05873_0000.png")
    if img_path.exists():
        img = Image.open(img_path).convert("RGB")
        dets = backend.predict(img, target_classes=["tennis-court"], confidence_threshold=0.35)
        assert len(dets) > 0
        assert all(d.class_name == "tennis-court" for d in dets)
    else:
        img = Image.new("RGB", (800, 800), color=(100, 100, 100))
        dets = backend.predict(img, target_classes=["storage_tank"], confidence_threshold=0.35)
        assert isinstance(dets, list)

    meta = backend.get_metadata()
    assert meta["model_id"] == "yolo_vhr_v1"
    assert meta["device"] == "cpu"


def test_model_loader_instantiates_detector():
    loader = ModelLoader()
    backend = loader.get_or_load("yolo_vhr_v1")
    assert backend is not None
    assert backend.is_loaded()
    assert isinstance(backend, RemoteSensingYOLOBackend)
