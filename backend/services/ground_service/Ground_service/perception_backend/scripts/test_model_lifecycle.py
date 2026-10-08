"""CLI test script for validating model lifecycle and execution workflows.

Supports:
- Full lifecycle sequence: POST /load -> POST /run -> POST /unload -> GET /health
- Reproducible automated smoke test: POST /smoke_test (load -> run -> unload)
- Generates structured artifacts under outputs/model_lifecycle/<timestamp>/
"""

from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any, Optional
import urllib.request
import urllib.error
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("test_model_lifecycle")


def clean_base64_from_dict(obj: Any) -> Any:
    """Recursively strip large image_base64 values from artifact dicts."""
    if isinstance(obj, dict):
        return {
            k: ("[BASE64_IMAGE_DATA_REDACTED]" if k in ("image_base64", "test_image_base64") and isinstance(v, str) and len(v) > 200 else clean_base64_from_dict(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [clean_base64_from_dict(item) for item in obj]
    return obj


class LifecycleClient:
    """Client for testing FastAPI perception lifecycle APIs."""

    def __init__(self, server_url: Optional[str] = None) -> None:
        self.server_url = server_url.rstrip("/") if server_url else None
        if not self.server_url:
            from fastapi.testclient import TestClient
            from app.api.app import app
            self.test_client = TestClient(app)
        else:
            self.test_client = None

    def post(self, endpoint: str, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        endpoint = "/" + endpoint.lstrip("/")
        if self.test_client:
            resp = self.test_client.post(endpoint, json=payload)
            return resp.status_code, resp.json()
        else:
            url = f"{self.server_url}{endpoint}"
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers={"Content-Type": "application/json", "Accept": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    body = r.read().decode("utf-8")
                    return r.status, json.loads(body)
            except urllib.error.HTTPError as he:
                body = he.read().decode("utf-8", errors="replace")
                try:
                    res = json.loads(body)
                except Exception:
                    res = {"detail": body}
                return he.code, res

    def get(self, endpoint: str) -> tuple[int, dict[str, Any]]:
        endpoint = "/" + endpoint.lstrip("/")
        if self.test_client:
            resp = self.test_client.get(endpoint)
            return resp.status_code, resp.json()
        else:
            url = f"{self.server_url}{endpoint}"
            req = urllib.request.Request(url, headers={"Accept": "application/json"}, method="GET")
            try:
                with urllib.request.urlopen(req, timeout=30) as r:
                    body = r.read().decode("utf-8")
                    return r.status, json.loads(body)
            except urllib.error.HTTPError as he:
                body = he.read().decode("utf-8", errors="replace")
                try:
                    res = json.loads(body)
                except Exception:
                    res = {"detail": body}
                return he.code, res


def main() -> int:
    parser = argparse.ArgumentParser(description="Test hosting-agnostic model lifecycle.")
    parser.add_argument("--model-id", default="remote_sam_sam3_v2", help="Model ID or tool name")
    parser.add_argument("--image", default="test_assets/default_satellite.png", help="Path to input satellite image")
    parser.add_argument("--query", default="Locate the swimming pool.", help="Prompt or referring expression")
    parser.add_argument("--provider", default=None, help="Explicit provider override (local, http, modal)")
    parser.add_argument("--smoke-test", action="store_true", help="Execute complete load -> run -> unload smoke test")
    parser.add_argument("--server-url", default=None, help="Remote server URL (e.g. http://localhost:8000 or Modal URL)")
    parser.add_argument("--output-dir", default=None, help="Base output directory for artifacts")
    args = parser.parse_args()

    # 1. Setup output artifact directory
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir or f"outputs/model_lifecycle/{timestamp}")
    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Initialized output artifact directory: {out_dir}")

    # 2. Read image
    img_path = Path(args.image)
    if not img_path.exists():
        fallback_path = Path("examples/real_satellite/05873_0000.png")
        if fallback_path.exists():
            img_path = fallback_path
        else:
            logger.error(f"Image file '{img_path}' not found.")
            return 1

    with Image.open(img_path) as im:
        img_w, img_h = im.size
    with open(img_path, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("ascii")

    client = LifecycleClient(server_url=args.server_url)
    model_id = args.model_id
    query = args.query

    metadata: dict[str, Any] = {
        "timestamp": timestamp,
        "model_id": model_id,
        "image": str(img_path),
        "image_dimensions": [img_w, img_h],
        "query": query,
        "mode": "smoke_test" if args.smoke_test else "standard_lifecycle",
        "server_url": args.server_url or "in-process",
        "operations": [],
    }

    if args.smoke_test:
        logger.info(f"=== Starting POST /smoke_test for '{model_id}' ===")
        t0 = time.perf_counter()
        code, smoke_res = client.post(
            "/smoke_test",
            payload={"model_id": model_id, "image_base64": img_b64, "query": query},
        )
        latency = (time.perf_counter() - t0) * 1000.0

        metadata["operations"].append({
            "operation": "smoke_test",
            "status_code": code,
            "latency_ms": round(latency, 2),
            "status": smoke_res.get("status", "error"),
        })

        with open(out_dir / "smoke_test.json", "w", encoding="utf-8") as f:
            json.dump(clean_base64_from_dict(smoke_res), f, indent=2)

        # Also capture health state
        _, health_res = client.get("/health")
        with open(out_dir / "health.json", "w", encoding="utf-8") as f:
            json.dump(health_res, f, indent=2)

        metadata["smoke_test_outcome"] = smoke_res.get("status")
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"POST /smoke_test finished with HTTP {code} (Latency: {latency:.2f}ms)")
        print("\n--- Smoke Test Response ---")
        print(json.dumps(clean_base64_from_dict(smoke_res), indent=2))
        return 0 if code == 200 and smoke_res.get("status") == "passed" else 1

    # Standard Lifecycle Workflow: LOAD -> RUN -> UNLOAD -> HEALTH
    logger.info(f"=== Step 1: POST /load for '{model_id}' ===")
    t0 = time.perf_counter()
    code, load_res = client.post(
        "/load",
        payload={
            "model_id": model_id,
            "run_startup_test": True,
            "provider": args.provider,
        },
    )
    load_lat = (time.perf_counter() - t0) * 1000.0
    logger.info(f"POST /load completed: HTTP {code} in {load_lat:.2f}ms")
    with open(out_dir / "load.json", "w", encoding="utf-8") as f:
        json.dump(clean_base64_from_dict(load_res), f, indent=2)

    metadata["operations"].append({
        "operation": "load",
        "status_code": code,
        "latency_ms": round(load_lat, 2),
        "model_instance_id": load_res.get("model_instance_id"),
        "real_model": load_res.get("real_model"),
        "device": load_res.get("device"),
        "dtype": load_res.get("dtype"),
        "checkpoint": load_res.get("checkpoint"),
    })
    print("\n--- Load Response ---")
    print(json.dumps(clean_base64_from_dict(load_res), indent=2))

    if code != 200:
        logger.error("POST /load failed! Aborting lifecycle test.")
        with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        return 1

    inst_id = load_res.get("model_instance_id")

    logger.info(f"=== Step 2: POST /run for '{model_id}' (instance: {inst_id}) ===")
    t0 = time.perf_counter()
    run_payload = {
        "model_id": model_id,
        "model_instance_id": inst_id,
        "image_base64": img_b64,
        "query": query,
    }
    if "ship" in query.lower() or "tennis" in query.lower() or "vehicle" in query.lower():
        run_payload["target_classes"] = [query.strip().lower()]

    code, run_res = client.post("/run", payload=run_payload)
    run_lat = (time.perf_counter() - t0) * 1000.0
    logger.info(f"POST /run completed: HTTP {code} in {run_lat:.2f}ms")
    with open(out_dir / "run.json", "w", encoding="utf-8") as f:
        json.dump(clean_base64_from_dict(run_res), f, indent=2)

    metadata["operations"].append({
        "operation": "run",
        "status_code": code,
        "latency_ms": round(run_lat, 2),
        "detections_count": run_res.get("count"),
        "groundings_count": run_res.get("groundings_count"),
        "real_model": run_res.get("real_model"),
    })
    print("\n--- Run Response Snippet ---")
    run_snippet = dict(run_res)
    if "groundings" in run_snippet and run_snippet["groundings"]:
        run_snippet["groundings"] = run_snippet["groundings"][:3]
    if "detections" in run_snippet and run_snippet["detections"]:
        run_snippet["detections"] = run_snippet["detections"][:3]
    print(json.dumps(clean_base64_from_dict(run_snippet), indent=2))

    if code != 200:
        logger.error("POST /run failed!")

    logger.info(f"=== Step 3: POST /unload for '{model_id}' ===")
    t0 = time.perf_counter()
    code, unload_res = client.post("/unload", payload={"model_id": model_id})
    unload_lat = (time.perf_counter() - t0) * 1000.0
    logger.info(f"POST /unload completed: HTTP {code} in {unload_lat:.2f}ms")
    with open(out_dir / "unload.json", "w", encoding="utf-8") as f:
        json.dump(unload_res, f, indent=2)

    metadata["operations"].append({
        "operation": "unload",
        "status_code": code,
        "latency_ms": round(unload_lat, 2),
        "memory_release": unload_res.get("memory_release"),
    })
    print("\n--- Unload Response ---")
    print(json.dumps(unload_res, indent=2))

    logger.info("=== Step 4: GET /health ===")
    code, health_res = client.get("/health")
    with open(out_dir / "health.json", "w", encoding="utf-8") as f:
        json.dump(health_res, f, indent=2)

    metadata["final_health"] = health_res
    with open(out_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n--- Final Health State ---")
    print(json.dumps(health_res, indent=2))
    logger.info(f"Full lifecycle validation completed. Artifacts saved in: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
