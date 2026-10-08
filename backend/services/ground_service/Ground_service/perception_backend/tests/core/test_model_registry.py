"""Unit tests for ModelRegistry thread-safe state machine."""

import concurrent.futures
import pytest
from app.core.model_registry import ModelRegistry, ModelState


def test_registry_initial_state():
    reg = ModelRegistry()
    sess = reg.get_or_create("test_model", provider="local")
    assert sess.state == ModelState.UNLOADED
    assert not reg.is_loaded("test_model")


def test_registry_lifecycle_transitions():
    reg = ModelRegistry()
    m_id = "remote_sam_sam3_v2"

    # 1. Loading
    s1 = reg.set_loading(m_id, provider="local")
    assert s1.state == ModelState.LOADING
    assert not reg.is_loaded(m_id)

    # 2. Loaded
    inst_id = "inst_test_001"
    s2 = reg.set_loaded(m_id, provider="local", instance_id=inst_id, metadata={"device": "cuda", "real_model": True})
    assert s2.state == ModelState.LOADED
    assert reg.is_loaded(m_id)
    assert s2.model_instance_id == inst_id
    assert reg.get_by_instance_id(inst_id) is not None

    # 3. Running
    s3 = reg.set_running(m_id)
    assert s3.state == ModelState.RUNNING
    assert reg.is_loaded(m_id)

    # 4. Finished Run -> back to LOADED
    s4 = reg.set_run_finished(m_id)
    assert s4.state == ModelState.LOADED
    assert reg.is_loaded(m_id)

    # 5. Unloading
    s5 = reg.set_unloading(m_id)
    assert s5.state == ModelState.UNLOADING

    # 6. Unloaded
    s6 = reg.set_unloaded(m_id)
    assert s6.state == ModelState.UNLOADED
    assert not reg.is_loaded(m_id)
    assert s6.model_instance_id is None
    assert reg.get_by_instance_id(inst_id) is None


def test_registry_error_transition():
    reg = ModelRegistry()
    m_id = "yolo_fail"
    reg.set_loading(m_id, provider="http")
    reg.set_error(m_id, "Connection refused")
    sess = reg.get(m_id)
    assert sess.state == ModelState.ERROR
    assert "Connection refused" in sess.error_message
    assert not reg.is_loaded(m_id)


def test_registry_thread_safety():
    reg = ModelRegistry()
    m_id = "concurrent_model"

    def worker(worker_id: int):
        reg.set_loading(m_id, provider=f"prov_{worker_id}")
        reg.set_loaded(m_id, provider=f"prov_{worker_id}", instance_id=f"inst_{worker_id}")
        reg.set_running(m_id)
        reg.set_run_finished(m_id)
        reg.set_unloaded(m_id)
        return True

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(worker, i) for i in range(20)]
        results = [f.result() for f in futures]

    assert all(results)
    assert not reg.is_loaded(m_id)
