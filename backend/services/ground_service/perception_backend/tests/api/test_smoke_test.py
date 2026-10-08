"""API tests for POST /smoke_test and GET /health."""

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


def test_health_endpoint_before_load():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["service"] == "sac-geoai-perception"
    assert "models" in data
    # No credentials or secrets exposed
    assert "api_key" not in str(data).lower()
    assert "token" not in str(data).lower()


def test_smoke_test_full_lifecycle_success():
    model_id = "remote_sam_sam3_v2"

    fake_load_res = {
        "status": "loaded",
        "model_id": model_id,
        "model_instance_id": "inst_smoke_001",
        "provider": "local",
        "real_model": True,
        "load_latency_ms": 110.0,
    }
    fake_run_res = {
        "groundings": [
            {
                "object_id": "g_01",
                "class_name": "pool",
                "confidence": {"score": 0.95, "calibrated": True, "source": "real_model"},
                "bbox_px": {"x_min": 10.0, "y_min": 10.0, "x_max": 50.0, "y_max": 50.0},
            }
        ],
        "groundings_count": 1,
        "model_metadata": {"real_model": True},
        "latency_ms": 45.0,
    }
    fake_unload_res = {
        "status": "unloaded",
        "model_id": model_id,
        "model_instance_id": "inst_smoke_001",
        "unload_latency_ms": 10.0,
        "memory_release": "150 MB",
    }

    with patch("app.core.providers.local_provider.LocalProvider.load", return_value=fake_load_res), \
         patch("app.core.providers.local_provider.LocalProvider.run", return_value=fake_run_res), \
         patch("app.core.providers.local_provider.LocalProvider.unload", return_value=fake_unload_res):

        response = client.post("/smoke_test", json={"model_id": model_id})
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "passed"
        assert data["model_id"] == model_id
        assert data["real_model"] is True
        assert data["load"]["status"] == "loaded"
        assert data["run"]["status"] == "success"
        assert data["unload"]["status"] == "unloaded"

        # Crucial requirement: /smoke_test must NOT leave model loaded
        assert model_registry.is_loaded(model_id) is False


def test_smoke_test_fails_on_run_error():
    model_id = "yolo_obb_v1"

    fake_load_res = {
        "status": "loaded",
        "model_id": model_id,
        "model_instance_id": "inst_smoke_fail",
        "provider": "local",
        "real_model": True,
        "load_latency_ms": 50.0,
    }

    with patch("app.core.providers.local_provider.LocalProvider.load", return_value=fake_load_res), \
         patch("app.core.providers.local_provider.LocalProvider.run", side_effect=RuntimeError("GPU OOM")):

        response = client.post("/smoke_test", json={"model_id": model_id})
        assert response.status_code == 500
        # Model should still be unloaded on error cleanup or left in error state
