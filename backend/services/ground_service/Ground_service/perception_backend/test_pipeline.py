"""Turnkey verification script for the Perception Backend.

Tests:
1. GET /health
2. POST /smoke_test for object_detector (yolo_obb_v1)
3. Full lifecycle for segmentation_grounding (remote_sam_sam3_v2): LOAD -> RUN -> UNLOAD
4. Full lifecycle for image_analyzer (earthmind_v1): LOAD -> RUN -> UNLOAD
5. Zero-Mock Policy & Real Model Guarantee assertions
"""

import base64
import os
import sys
from pathlib import Path

# Add paths
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "packages" / "sac-shared-contracts" / "src"))

# Change cwd to BASE_DIR so relative config and weights paths resolve cleanly
os.chdir(str(BASE_DIR))

from fastapi.testclient import TestClient
from app.api.app import app

def run_tests():
    client = TestClient(app)
    image_path = BASE_DIR / "test_assets" / "default_satellite.png"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing test asset at {image_path}")

    with open(image_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    print("=" * 65)
    print("SAC GeoAI Perception Backend - Comprehensive Verification")
    print("=" * 65)

    # 1. Health check
    print("\n[1/4] Testing GET /health...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    health_data = resp.json()
    print(f"  Service: {health_data.get('service')}")
    print(f"  Status:  {health_data.get('status')}")
    print(f"  Models:  {list(health_data.get('models', {}).keys())}")
    print("  --> PASS")

    # 2. Smoke test for detection (yolo_obb_v1)
    print("\n[2/4] Testing POST /smoke_test for 'yolo_obb_v1' (Rotated YOLO-MS / OBB)...")
    resp = client.post("/smoke_test", json={"model_id": "yolo_obb_v1"})
    assert resp.status_code == 200, f"Smoke test failed: {resp.text}"
    smoke_data = resp.json()
    assert smoke_data.get("status") == "passed", f"Smoke test did not pass: {smoke_data}"
    assert smoke_data.get("real_model") is True, "Mock backend detected in smoke test!"
    print(f"  Status:        {smoke_data.get('status')}")
    print(f"  Real Model:    {smoke_data.get('real_model')}")
    print(f"  Provider:      {smoke_data.get('provider')}")
    print(f"  Load Latency:  {smoke_data.get('load', {}).get('latency_ms', 0):.1f}ms")
    print(f"  Run Latency:   {smoke_data.get('run', {}).get('latency_ms', 0):.1f}ms")
    print(f"  Unload Latency:{smoke_data.get('unload', {}).get('latency_ms', 0):.1f}ms")
    print("  --> PASS")

    # 3. Lifecycle for segmentation (remote_sam_sam3_v2 / SegEarth-OV3)
    print("\n[3/4] Testing Lifecycle for 'remote_sam_sam3_v2' (SegEarth-OV3 / SAM3)...")
    # Load
    load_resp = client.post("/load", json={"model_id": "remote_sam_sam3_v2"})
    assert load_resp.status_code == 200, f"Load failed: {load_resp.text}"
    load_json = load_resp.json()
    assert load_json.get("status") == "loaded"
    assert load_json.get("real_model") is True
    print(f"  [Load] Model instance ID: {load_json.get('model_instance_id')}")
    print(f"  [Load] Device: {load_json.get('device')}, Latency: {load_json.get('load_latency_ms', 0):.1f}ms")

    # Run
    run_resp = client.post("/run", json={
        "model_id": "remote_sam_sam3_v2",
        "image_base64": img_b64,
        "query": "runway, airplane",
        "target_classes": ["runway", "airplane"]
    })
    assert run_resp.status_code == 200, f"Run failed: {run_resp.text}"
    run_json = run_resp.json()
    assert run_json.get("status") == "success"
    assert run_json.get("real_model") is True
    assert "mock" not in str(run_json.get("model_metadata", {}).get("source_model", "")).lower()
    print(f"  [Run] Success! Groundings count: {run_json.get('groundings_count')}")
    print(f"  [Run] Real Model: {run_json.get('real_model')}, Latency: {run_json.get('latency_ms', 0):.1f}ms")

    # Unload
    unload_resp = client.post("/unload", json={"model_id": "remote_sam_sam3_v2"})
    assert unload_resp.status_code == 200, f"Unload failed: {unload_resp.text}"
    print(f"  [Unload] Unloaded successfully in {unload_resp.json().get('unload_latency_ms', 0):.1f}ms.")
    print("  --> PASS")

    # 4. Lifecycle for image analyzer (earthmind_v1)
    print("\n[4/4] Testing Lifecycle for 'earthmind_v1' (Image Analyzer)...")
    # Load
    load_resp = client.post("/load", json={"model_id": "earthmind_v1"})
    assert load_resp.status_code == 200, f"Load failed: {load_resp.text}"
    load_json = load_resp.json()
    assert load_json.get("status") == "loaded"

    # Run
    run_resp = client.post("/run", json={
        "model_id": "earthmind_v1",
        "image_base64": img_b64,
        "query": "Describe the airport infrastructure and runways in this satellite view."
    })
    assert run_resp.status_code == 200, f"Run failed: {run_resp.text}"
    run_json = run_resp.json()
    assert run_json.get("status") == "success"
    assert run_json.get("real_model") is True
    caption = run_json.get("response", {}).get("caption") or run_json.get("response", {}).get("observations", [""])[0]
    print(f"  [Run] Success! Caption: {str(caption)[:85]}...")
    print(f"  [Run] Real Model: {run_json.get('real_model')}")

    # Unload
    unload_resp = client.post("/unload", json={"model_id": "earthmind_v1"})
    assert unload_resp.status_code == 200, f"Unload failed: {unload_resp.text}"
    print(f"  [Unload] Unloaded successfully in {unload_resp.json().get('unload_latency_ms', 0):.1f}ms.")
    print("  --> PASS")

    print("\n" + "=" * 65)
    print("ALL VERIFICATION CHECKS PASSED WITH 100% REAL MODEL GUARANTEE!")
    print("=" * 65)

if __name__ == "__main__":
    run_tests()
