"""
Manual Real-Inference Test Script for Deployed EarthDial Temporal Comparison Endpoint
=====================================================================================

Performs real end-to-end inference against the live Modal deployment using actual
temporal images.
"""

import base64
import json
import sys
import time
from pathlib import Path
import requests

# Base URL for the deployed service on Modal
DEFAULT_ENDPOINT = "https://me240003014--earthdial-temporal-service-earthdialmodelse-66ac19.modal.run"


def encode_image_to_base64(img_path: Path) -> str:
    with open(img_path, "rb") as f:
        data = f.read()
    ext = img_path.suffix.lstrip(".").lower()
    if ext == "jpg":
        ext = "jpeg"
    return f"data:image/{ext};base64,{base64.b64encode(data).decode('utf-8')}"


def test_endpoint(base_url: str = DEFAULT_ENDPOINT):
    base_url = base_url.rstrip("/")
    print("=" * 70)
    print(" EarthDial Temporal Comparison - Live Endpoint Testing")
    print(f" Target URL: {base_url}")
    print("=" * 70)

    # 1. Health check
    print("\n[Step 1] Checking /health endpoint...")
    t0 = time.time()
    try:
        resp = requests.get(f"{base_url}/health", timeout=300)
        resp.raise_for_status()
        health_data = resp.json()
        print(f"  Status Code: {resp.status_code}")
        print(f"  Health Info: {json.dumps(health_data, indent=2)}")
        print(f"  Cold-start / Response Time: {round(time.time() - t0, 2)}s")
    except Exception as e:
        print(f"  Health check failed: {e}")
        return False

    # 2. Get prompt templates
    print("\n[Step 2] Fetching recommended /prompts...")
    try:
        resp = requests.get(f"{base_url}/prompts", timeout=30)
        templates = resp.json().get("templates", [])
        print(f"  Available prompt templates: {len(templates)}")
        for t in templates[:2]:
            print(f"   - [{t['task_category']}] {t['name']}: '{t['query'][:60]}...'")
    except Exception as e:
        print(f"  Prompts fetch failed: {e}")

    # 3. Load existing images from test_data
    test_data_dir = Path(__file__).resolve().parent / "test_data"
    
    # Check for satellite-image-t1.png (hyphenated or space separated)
    img1 = test_data_dir / "satellite-image-t1.png"
    if not img1.exists():
        img1 = test_data_dir / "sattelite image t1.png"
    if not img1.exists():
        img1 = test_data_dir / "satellite image t1.png"

    img2 = test_data_dir / "satellite-image-t2.png"
    if not img2.exists():
        img2 = test_data_dir / "satellite image t2.png"

    if not img1.exists() or not img2.exists():
        raise FileNotFoundError(
            f"Required test images not found in {test_data_dir}. "
            f"Found: {[f.name for f in test_data_dir.iterdir()]}"
        )

    print(f"\n[Step 3] Preparing temporal image pair for real inference:")
    print(f"  T1 Image: {img1} ({img1.stat().st_size} bytes)")
    print(f"  T2 Image: {img2} ({img2.stat().st_size} bytes)")

    b64_img1 = encode_image_to_base64(img1)
    b64_img2 = encode_image_to_base64(img2)

    # 4. Execute Real Temporal Comparison Inference via POST /compare
    print("\n[Step 4] Executing Real Inference on EarthDial (POST /compare)...")
    payload = {
        "images": [b64_img1, b64_img2],
        "query": "Are there any semantic and structural changes detected between the provided images?",
        "prompt_style": "evaluation",
        "task_token": "[changedet] [hr_rgb_temp_0.5]",
        "temperature": 0.0,
        "max_new_tokens": 150,
        "dynamic_image_size": True
    }

    t_start = time.time()
    try:
        resp = requests.post(f"{base_url}/compare", json=payload, timeout=300)
        elapsed = round(time.time() - t_start, 2)
        resp.raise_for_status()
        result = resp.json()

        print("\n" + "=" * 70)
        print(" EARTHDIAL REAL INFERENCE RESULT:")
        print("=" * 70)
        print(f"Query           : {result.get('query')}")
        print(f"Formatted Prompt: {result.get('formatted_prompt')}")
        print(f"\nModel Output:\n{result.get('response')}\n")
        print("-" * 70)
        print(f"Device Used     : {result.get('device')}")
        print(f"Inference Latency: {result.get('execution_time_seconds')}s (Network total: {elapsed}s)")
        print(f"Image Metadata  : {result.get('image_metadata')}")
        print("=" * 70)
        return True

    except Exception as e:
        print(f"  Inference request failed: {e}")
        if 'resp' in locals() and hasattr(resp, "text"):
            print(f"  Response text: {resp.text}")
        return False


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_ENDPOINT
    success = test_endpoint(target)
    sys.exit(0 if success else 1)
