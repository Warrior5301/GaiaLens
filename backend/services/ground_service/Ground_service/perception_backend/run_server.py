"""Entrypoint to launch the SAC GeoAI Perception FastAPI service locally."""

import os
import sys
from pathlib import Path

# Add perception_backend and packages to sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "packages" / "sac-shared-contracts" / "src"))

import uvicorn

if __name__ == "__main__":
    print(f"Starting SAC GeoAI Perception Service from {BASE_DIR}...")
    print("API Documentation: http://localhost:8000/docs")
    print("Health Endpoint:   http://localhost:8000/health")
    uvicorn.run("app.api.app:app", host="0.0.0.0", port=8000, reload=False)
