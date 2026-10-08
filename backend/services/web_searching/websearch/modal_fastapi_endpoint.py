"""Expose the existing location-context FastAPI app as a public Modal service.

Deploy from this folder with:
    modal deploy modal_fastapi_endpoint.py
"""

from __future__ import annotations

import modal
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REMOTE_ROOT = "/root/location_context_project"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .add_local_file(
        str(ROOT / "location_context_api.py"),
        remote_path=f"{REMOTE_ROOT}/location_context_api.py",
    )
    .add_local_file(
        str(ROOT / "location_context_pipeline.py"),
        remote_path=f"{REMOTE_ROOT}/location_context_pipeline.py",
    )
    .add_local_file(
        str(ROOT / "geoclip_client.py"),
        remote_path=f"{REMOTE_ROOT}/geoclip_client.py",
    )
)

app = modal.App("location-context-fastapi")
earth_engine_secret = modal.Secret.from_name("earth-engine")


@app.function(
    image=image,
    gpu="T4",
    timeout=1800,
    secrets=[earth_engine_secret],
)
@modal.asgi_app()
def fastapi_app():
    """Initialize Earth Engine and serve the existing FastAPI application."""
    import json
    import os
    import sys

    project = os.environ.get("EE_PROJECT")
    credential_json = os.environ.get("EE_SERVICE_ACCOUNT_JSON")
    if not project:
        raise RuntimeError("Modal secret 'earth-engine' must contain EE_PROJECT.")
    if not credential_json:
        raise RuntimeError(
            "Modal secret 'earth-engine' must contain EE_SERVICE_ACCOUNT_JSON."
        )

    try:
        credential_data = json.loads(credential_json)
        client_email = credential_data["client_email"]
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "EE_SERVICE_ACCOUNT_JSON must contain a valid service-account JSON key."
        ) from exc

    credential_path = Path("/tmp/earth-engine-service-account.json")
    credential_path.write_text(credential_json, encoding="utf-8")
    credential_path.chmod(0o600)

    import ee

    credentials = ee.ServiceAccountCredentials(
        client_email,
        key_file=str(credential_path),
    )
    ee.Initialize(credentials, project=project)

    sys.path.insert(0, REMOTE_ROOT)
    from location_context_api import app as fastapi_application
    from location_context_api import explorer

    # Earth Engine is initialized above with the Modal service-account secret.
    explorer.ee.project = project
    explorer.ee._initialized = True
    return fastapi_application
