"""Expose the GSD FastAPI service as a Modal HTTPS endpoint.

Deploy from this directory with:
    modal deploy modal_app.py
"""

from pathlib import Path

import modal

BASE_DIR = Path(__file__).resolve().parent
app = modal.App("gsd-service-api")

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install_from_requirements(str(BASE_DIR / "requirements.txt"))
    .add_local_dir(str(BASE_DIR), remote_path="/root/gsd_dataset", copy=True)
)


@app.function(
    image=image,
    timeout=600,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    """Serve the GSD FastAPI app on Modal."""
    import os
    import sys

    service_dir = "/root/gsd_dataset"
    if service_dir not in sys.path:
        sys.path.insert(0, service_dir)
    os.chdir(service_dir)

    from main import app as web_app

    return web_app
