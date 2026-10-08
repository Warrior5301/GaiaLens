"""
EarthDial REST API - Client Example
====================================

Demonstrates how to send requests to the EarthDial FastAPI service using Python's `requests`.

Prerequisites:
    Start the API server first:
        python run.py --port 8000
    (Add --mock if testing locally without GPU weights)
"""

import sys
from pathlib import Path
import requests

API_BASE_URL = "http://localhost:8000"

# Sample images
test_dir = Path(__file__).resolve().parent / "test_data"
img1 = test_dir / "satellite-image-t1.png"
img2 = test_dir / "satellite-image-t2.png"

# Fallback if specific test files are missing
if not img1.exists() or not img2.exists():
    img1 = test_dir / "satellite_t1_baseline.png"
    img2 = test_dir / "satellite_t2_modified.png"


def test_health():
    print("--- 1. Health & Status Check ---")
    resp = requests.get(f"{API_BASE_URL}/health")
    print(f"Status: {resp.status_code}")
    print(resp.json())
    print()


def test_model_info():
    print("--- 2. Model Architecture & Info ---")
    resp = requests.get(f"{API_BASE_URL}/model/info")
    print(f"Status: {resp.status_code}")
    print(resp.json())
    print()


def test_temporal_comparison_json():
    print("--- 3. Temporal Comparison via JSON (/compare) ---")
    payload = {
        "images": [str(img1), str(img2)],
        "query": "Are there any semantic changes detected between the two satellite observations?",
        "prompt_style": "evaluation",
        "temperature": 0.0,
        "max_new_tokens": 128
    }
    resp = requests.post(f"{API_BASE_URL}/compare", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(f"Model Response: {data['response']}")
        print(f"Latency       : {data['execution_time_seconds']}s")
        print(f"Device        : {data['device']}")
    else:
        print("Error:", resp.text)
    print()


def test_temporal_comparison_upload():
    print("--- 4. Temporal Comparison via File Upload (/compare/upload) ---")
    with open(img1, "rb") as f1, open(img2, "rb") as f2:
        files = [
            ("files", ("image_t1.png", f1, "image/png")),
            ("files", ("image_t2.png", f2, "image/png"))
        ]
        data = {
            "query": "Identify any newly constructed roads or urban expansion.",
            "prompt_style": "evaluation",
            "temperature": 0.0
        }
        resp = requests.post(f"{API_BASE_URL}/compare/upload", files=files, data=data)
        print(f"Status: {resp.status_code}")
        if resp.ok:
            result = resp.json()
            print(f"Model Response: {result['response']}")
            print(f"Latency       : {result['execution_time_seconds']}s")
        else:
            print("Error:", resp.text)
    print()


def test_model_chat():
    print("--- 5. Direct Model Chat / VQA (/model/chat) ---")
    payload = {
        "images": [str(img1)],
        "prompt": "<image>\nDescribe the geographic terrain and land features in this observation.",
        "max_new_tokens": 100,
        "temperature": 0.0
    }
    resp = requests.post(f"{API_BASE_URL}/model/chat", json=payload)
    print(f"Status: {resp.status_code}")
    if resp.ok:
        data = resp.json()
        print(f"Model Response: {data['response']}")
        print(f"Latency       : {data['execution_time_seconds']}s")
    else:
        print("Error:", resp.text)
    print()


if __name__ == "__main__":
    try:
        test_health()
        test_model_info()
        test_temporal_comparison_json()
        test_temporal_comparison_upload()
        test_model_chat()
    except requests.exceptions.ConnectionError:
        print(f"[!] Could not connect to API server at {API_BASE_URL}.")
        print("    Please start the server first using:")
        print("    python run.py --port 8000 --mock")
