"""Modal deployment for SAC GeoAI Phase 4 Perception Tools.

Deploys:
- Object_Detector
- Segmentation_Grounding
- Image_Analyzer

Complies with sac-shared-contracts and Phase 4 standards.
"""

from __future__ import annotations

import base64
from io import BytesIO
import os
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent.parent.parent
REMOTE = "/root/sac-geoai"

image = (
    modal.Image.from_registry("pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime")
    .apt_install("libgl1", "libglib2.0-0", "libxcb1", "libxrender1", "libxext6")
    .pip_install(
        "fastapi",
        "python-multipart",
        "transformers>=4.48.0,<5.0.0",
        "pydantic>=2.7.0",
        "numpy>=1.26.0",
        "pillow>=10.0.0",
        "pyyaml>=6.0.1",
        "scipy>=1.11.0",
        "scikit-learn>=1.3.0",
        "shapely>=2.0.0",
        "json-repair>=0.30",
        "ultralytics>=8.0.0",
        "opencv-python-headless",
    )
    .add_local_dir(str(ROOT / "app"), remote_path=f"{REMOTE}/app", copy=True)
    .add_local_dir(str(ROOT / "config"), remote_path=f"{REMOTE}/config", copy=True)
    .add_local_dir(str(ROOT / "weights"), remote_path=f"{REMOTE}/weights", copy=True)
    .add_local_dir(str(ROOT / "test_assets"), remote_path=f"{REMOTE}/test_assets", copy=True)
    .add_local_dir(
        str(ROOT / "packages/sac-shared-contracts"),
        remote_path=f"{REMOTE}/packages/sac-shared-contracts",
        copy=True,
    )
    .run_commands(
        f"pip install -e {REMOTE}/packages/sac-shared-contracts",
    )
    .env({"PYTHONPATH": REMOTE})
)

app = modal.App("sac-geoai-perception", image=image)


@app.function(gpu="T4", timeout=600)
@modal.asgi_app()
def api():
    """ASGI application serving hosting-agnostic FastAPI control layer."""
    os.chdir(REMOTE)
    from app.api.app import app as fastapi_app
    return fastapi_app

