"""API tests for POST /run and tool-level run endpoints."""

import base64
from io import BytesIO
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from PIL import Image
import pytest

from app.api.app import app
from app.core.model_registry import model_registry

client = TestClient(app)


def create_dummy_png_base64() -> str:
    """Generate minimal valid 10x10 PNG base64 string for testing."""
    img = Image.new("RGB", (10, 10), color="blue")
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.fixture(autouse=True)
def clean_registry():
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)
    yield
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)


def test_run_before_load_returns_409():
    payload = {
        "model_id": "remote_sam_sam3_v2",
        "image_base64": create_dummy_png_base64(),
        "query": "Locate the swimming pool.",
    }
    response = client.post("/run", json=payload)
    assert response.status_code == 409
    data = response.json()
    assert data["status"] == "model_not_loaded"
    assert "Call POST /load before /run" in data["message"]


def test_run_invalid_base64_returns_400():
    # Pre-mark model as loaded
    model_registry.set_loaded("remote_sam_sam3_v2", provider="local", instance_id="inst_001")

    payload = {
        "model_id": "remote_sam_sam3_v2",
        "image_base64": "not_a_valid_base64!@@#$%",
        "query": "Locate lake.",
    }
    response = client.post("/run", json=payload)
    assert response.status_code == 400


def test_run_success_when_loaded():
    model_id = "yolo_obb_v1"
    model_registry.set_loaded(model_id, provider="local", instance_id="inst_det_001")

    fake_run_res = {
        "detections": [
            {
                "object_id": "det_001",
                "class_name": "ship",
                "confidence": {"score": 0.95, "calibrated": True, "source": "real_detector"},
                "bbox_px": {"x_min": 1.0, "y_min": 1.0, "x_max": 8.0, "y_max": 8.0},
            }
        ],
        "count": 1,
        "model_metadata": {"real_model": True, "model_id": model_id, "device": "cuda"},
        "latency_ms": 42.0,
    }

    with patch("app.core.providers.local_provider.LocalProvider.run", return_value=fake_run_res):
        payload = {
            "model_id": model_id,
            "image_base64": create_dummy_png_base64(),
            "target_classes": ["ship"],
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["count"] == 1
        assert len(data["detections"]) == 1
        assert data["real_model"] is True


def test_run_rejects_mock_backend_with_503():
    model_id = "remote_sam_sam3_v2"
    model_registry.set_loaded(model_id, provider="local", instance_id="inst_mock_001")

    # Provider returning forbidden mock
    fake_run_res = {
        "groundings": [],
        "model_metadata": {"real_model": True, "backend": "MockRemoteSAMBackend"},
    }

    with patch("app.core.providers.local_provider.LocalProvider.run", return_value=fake_run_res):
        payload = {
            "model_id": model_id,
            "image_base64": create_dummy_png_base64(),
            "query": "pool",
        }
        response = client.post("/run", json=payload)
        assert response.status_code == 503
        data = response.json()
        assert data["status"] == "provider_unavailable"


def test_tool_specific_run():
    model_id = "yolo_obb_v1"
    model_registry.set_loaded(model_id, provider="local", instance_id="inst_tool_run")

    fake_run_res = {
        "detections": [],
        "count": 0,
        "model_metadata": {"real_model": True},
        "latency_ms": 20.0,
    }

    with patch("app.core.providers.local_provider.LocalProvider.run", return_value=fake_run_res):
        payload = {
            "image_base64": create_dummy_png_base64(),
            "target_classes": ["ship"],
        }
        response = client.post("/tools/object_detector/run", json=payload)
        assert response.status_code == 200
        assert response.json()["model_id"] == "yolo_obb_v1"
