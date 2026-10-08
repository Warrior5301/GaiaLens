"""API tests for POST /load and tool-level load endpoints."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from app.api.app import app
from app.core.model_registry import model_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure clean registry state before and after each test."""
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)
    yield
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)


def test_load_success():
    fake_load_res = {
        "status": "loaded",
        "model_id": "remote_sam_sam3_v2",
        "model_instance_id": "inst_mock_123",
        "provider": "local",
        "real_model": True,
        "device": "cuda",
        "dtype": "bfloat16",
        "checkpoint": "weights/segearth_ov3/segearth_ov3.pt",
        "load_latency_ms": 150.0,
    }

    with patch("app.core.providers.local_provider.LocalProvider.load", return_value=fake_load_res):
        response = client.post("/load", json={"model_id": "remote_sam_sam3_v2", "run_startup_test": False})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "loaded"
        assert data["model_id"] == "remote_sam_sam3_v2"
        assert data["model_instance_id"] == "inst_mock_123"
        assert data["real_model"] is True
        assert model_registry.is_loaded("remote_sam_sam3_v2") is True


def test_load_with_startup_test():
    fake_load_res = {
        "status": "loaded",
        "model_id": "yolo_obb_v1",
        "model_instance_id": "inst_yolo_456",
        "provider": "local",
        "real_model": True,
        "load_latency_ms": 120.0,
    }
    fake_run_res = {
        "detections": [{"class_name": "ship", "confidence": 0.9}],
        "model_metadata": {"real_model": True},
        "latency_ms": 50.0,
    }

    with patch("app.core.providers.local_provider.LocalProvider.load", return_value=fake_load_res), \
         patch("app.core.providers.local_provider.LocalProvider.run", return_value=fake_run_res):
        response = client.post("/load", json={"model_id": "yolo_obb_v1", "run_startup_test": True})
        assert response.status_code == 200
        data = response.json()
        assert data["smoke_test"] is not None
        assert data["smoke_test"]["status"] == "passed"
        # Must stay loaded after /load
        assert model_registry.is_loaded("yolo_obb_v1") is True


def test_tool_specific_load():
    fake_load_res = {
        "status": "loaded",
        "model_id": "yolo_obb_v1",
        "model_instance_id": "inst_tool_det",
        "provider": "local",
        "real_model": True,
        "load_latency_ms": 110.0,
    }
    with patch("app.core.providers.local_provider.LocalProvider.load", return_value=fake_load_res):
        response = client.post("/tools/object_detector/load", json={})
        assert response.status_code == 200
        data = response.json()
        assert data["model_id"] == "yolo_obb_v1"
        assert model_registry.is_loaded("yolo_obb_v1") is True
