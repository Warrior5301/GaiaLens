"""Unit tests for ModelProvider implementations and Real Model Guarantee validation."""

import json
from unittest.mock import MagicMock, patch
import urllib.error
import pytest

from app.core.providers import (
    BaseProvider,
    HTTPProvider,
    MockDetectedError,
    ModalProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    validate_real_model_guarantee,
)


def test_real_model_guarantee_success():
    valid_meta = {
        "model_id": "segearth_ov3",
        "checkpoint": "weights/segearth_ov3/segearth_ov3.pt",
        "device": "cuda",
        "dtype": "bfloat16",
        "real_model": True,
    }
    assert validate_real_model_guarantee(valid_meta, "test_provider") is True


def test_real_model_guarantee_fails_on_false_flag():
    invalid_meta = {
        "model_id": "mock_model",
        "real_model": False,
    }
    with pytest.raises(MockDetectedError, match="reported real_model=false"):
        validate_real_model_guarantee(invalid_meta, "test_provider")


def test_real_model_guarantee_fails_on_mock_strings():
    for mock_name in ["MockRemoteSAMBackend", "MockSAM3Backend", "mock_detector", "synthetic_runner"]:
        meta = {
            "model_id": "model_test",
            "backend": mock_name,
            "real_model": True,
        }
        with pytest.raises(MockDetectedError, match="forbidden mock signature"):
            validate_real_model_guarantee(meta, "test_provider")


def test_http_provider_load_success():
    prov = HTTPProvider(provider_name="http")
    fake_config = {
        "model_id": "remote_sam_sam3_v2",
        "endpoints": {
            "load": "http://fake-host/load",
            "run": "http://fake-host/run",
            "unload": "http://fake-host/unload",
            "health": "http://fake-host/health",
        },
        "timeouts": {"load_timeout_sec": 5.0},
    }

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = json.dumps({
        "status": "loaded",
        "model_instance_id": "inst_http_001",
        "real_model": True,
        "device": "cuda",
        "checkpoint": "model.pt",
    }).encode("utf-8")
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        res = prov.load(fake_config)
        assert res["status"] == "loaded"
        assert res["model_instance_id"] == "inst_http_001"
        assert prov.is_loaded is True
        assert res["real_model"] is True


def test_http_provider_timeout_error():
    prov = HTTPProvider(provider_name="http")
    fake_config = {
        "model_id": "remote_sam_sam3_v2",
        "endpoints": {"load": "http://fake-host/load"},
        "timeouts": {"load_timeout_sec": 1.0},
    }

    with patch("urllib.request.urlopen", side_effect=TimeoutError("timed out")):
        with pytest.raises(ProviderTimeoutError, match="timed out"):
            prov.load(fake_config)


def test_http_provider_http_error_rejection():
    prov = HTTPProvider(provider_name="http")
    fake_config = {
        "model_id": "remote_sam_sam3_v2",
        "endpoints": {"load": "http://fake-host/load"},
    }

    err = urllib.error.HTTPError(
        url="http://fake-host/load",
        code=503,
        msg="Service Unavailable",
        hdrs={},
        fp=MagicMock(read=lambda: b'{"detail": "Backend unavailable"}'),
    )
    with patch("urllib.request.urlopen", side_effect=err):
        with pytest.raises(ProviderError) as exc_info:
            prov.load(fake_config)
        assert exc_info.value.status_code == 503


def test_modal_provider_run_requires_load():
    prov = ModalProvider(provider_name="modal")
    with pytest.raises(ProviderError, match="Model is not loaded"):
        prov.run({"image_base64": "dummy", "query": "test"})
