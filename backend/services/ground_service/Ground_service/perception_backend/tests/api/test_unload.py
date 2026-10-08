"""API tests for POST /unload and tool-level unload endpoints."""

from unittest.mock import patch
from fastapi.testclient import TestClient
import pytest

from app.api.app import app
from app.core.model_registry import model_registry

client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_registry():
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)
    yield
    for m_id in list(model_registry._sessions.keys()):
        model_registry.set_unloaded(m_id)


def test_unload_success():
    model_id = "remote_sam_sam3_v2"
    model_registry.set_loaded(model_id, provider="local", instance_id="inst_ul_001")
    assert model_registry.is_loaded(model_id) is True

    fake_unload_res = {
        "status": "unloaded",
        "model_id": model_id,
        "model_instance_id": "inst_ul_001",
        "unload_latency_ms": 15.0,
        "memory_release": "120.5 MB",
        "gpu_memory_released_mb": 120.5,
    }

    with patch("app.core.providers.local_provider.LocalProvider.unload", return_value=fake_unload_res):
        response = client.post("/unload", json={"model_id": model_id})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "unloaded"
        assert data["model_id"] == model_id
        assert model_registry.is_loaded(model_id) is False


def test_unload_when_already_unloaded():
    model_id = "yolo_obb_v1"
    response = client.post("/unload", json={"model_id": model_id})
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "unloaded"
    assert data["memory_release"] == "already_unloaded"


def test_tool_specific_unload():
    model_id = "yolo_obb_v1"
    model_registry.set_loaded(model_id, provider="local", instance_id="inst_ul_tool")

    with patch("app.core.providers.local_provider.LocalProvider.unload", return_value={"status": "unloaded"}):
        response = client.post("/tools/object_detector/unload", json={})
        assert response.status_code == 200
        assert response.json()["status"] == "unloaded"
        assert model_registry.is_loaded(model_id) is False
