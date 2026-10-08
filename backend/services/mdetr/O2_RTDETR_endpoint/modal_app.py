"""
Modal deployment for the O2-RTDETR Remote Sensing Detection API.

Deploy:
    modal deploy modal_app.py

One-shot test:
    modal run modal_app.py

Secrets required (set via `modal secret create o2-rtdetr-secrets ...` or the UI):
    None required by default – paths are baked into the image.

Environment variables you can override at deploy time:
    AI4RS_ROOT            – where ai4rs is cloned inside the container (default: /ai4rs)
    O2_RTDETR_CONFIG      – full path to the mmdet config .py
    O2_RTDETR_CHECKPOINT  – full path to the .pth checkpoint
    O2_RTDETR_DEVICE      – "cuda:0" (default) or "cpu"
    O2_RTDETR_SCORE_THRESHOLD – float, default 0.25
"""

import modal

# ---------------------------------------------------------------------------
# 1.  Container image
#     We start from the official PyTorch CUDA image and layer everything on top.
# ---------------------------------------------------------------------------

AI4RS_ROOT = "/ai4rs"          # path inside the container
CHECKPOINT_URL = ""            # set to a direct download URL if you want the
                               # image to bake the checkpoint in; leave empty
                               # to mount it via a Modal Volume instead.

image = (
    modal.Image.from_registry(
        "pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime",
        add_python="3.11",
    )
    # System deps needed by OpenMMLab / OpenCV
    .apt_install(
        "git",
        "libgl1",
        "libglib2.0-0",
        "libsm6",
        "libxext6",
        "libxrender1",
        "wget",
    )
    # Core Python deps
    # setuptools is required by some OpenMMLab packages (provides pkg_resources)
    .pip_install(
        "setuptools",
        "fastapi>=0.110",
        "uvicorn[standard]>=0.27",
        "pydantic>=2.6",
        "numpy",
        "Pillow",
        "opencv-python-headless>=4.8",
        "openmim",
    )
    # OpenMMLab stack – install in order: mmengine → mmcv → mmdet → mmrotate
    # All deps are installed via mim before the ai4rs editable install,
    # so we pass --no-deps to avoid pulling in the old mmcv-full package.
    .run_commands(
        "mim install mmengine",
        "mim install 'mmcv>=2.0.0'",
        "mim install 'mmdet>=3.0.0'",
        # mmrotate (ai4rs fork) – adjust the repo URL if you use a private fork
        f"git clone --depth 1 https://github.com/open-mmlab/mmrotate {AI4RS_ROOT}",
        f"pip install -e {AI4RS_ROOT} --no-deps",
    )
    # Set env vars BEFORE adding local files — Modal requires add_local_* to be last
    .env({"PYTHONPATH": "/", "AI4RS_ROOT": AI4RS_ROOT})
    # Copy the application code into the image (must be the final build step)
    .add_local_dir("app", "/app")
)

# ---------------------------------------------------------------------------
# 2.  Modal Volume (optional but recommended) – mount your checkpoint here
#     so you don't have to rebuild the image every time you update weights.
#
#     Create once:
#         modal volume create o2-rtdetr-checkpoints
#         modal volume put o2-rtdetr-checkpoints ./checkpoints/o2_rtdetr_r18vd_dota_epoch_72.pth /checkpoints/
# ---------------------------------------------------------------------------

volume = modal.Volume.from_name("o2-rtdetr-checkpoints", create_if_missing=True)
VOLUME_MOUNT = "/checkpoints"

# ---------------------------------------------------------------------------
# 3.  Modal App
# ---------------------------------------------------------------------------

app = modal.App(
    name="o2-rtdetr-api",
    image=image,
)

# ---------------------------------------------------------------------------
# 4.  ASGI endpoint  (modal serve / modal deploy)
# ---------------------------------------------------------------------------

@app.function(
    gpu="A10G",                        # change to "T4" or "A100" as needed
    memory=16384,                       # 16 GB RAM
    timeout=300,                        # seconds per request
    volumes={VOLUME_MOUNT: volume},
    # Pass checkpoint path via env so config.py picks it up automatically.
    # Adjust the filename to match the file you uploaded to the volume.
    secrets=[],                         # add modal.Secret.from_name("...") if needed
)
@modal.asgi_app()
def fastapi_app():
    import os
    # Tell the app where to find the checkpoint on the mounted volume
    os.environ.setdefault(
        "O2_RTDETR_CHECKPOINT",
        f"{VOLUME_MOUNT}/o2_rtdetr_r18vd_dota_epoch_72.pth",
    )
    # Config lives inside the cloned ai4rs repo baked into the image
    os.environ.setdefault(
        "O2_RTDETR_CONFIG",
        f"{AI4RS_ROOT}/projects/rotated_rtdetr/configs/o2_rtdetr_r18vd_2xb4_72e_dota.py",
    )

    from app.main import app as _fastapi_app  # noqa: PLC0415
    return _fastapi_app


# ---------------------------------------------------------------------------
# 5.  Local test entry-point  (`modal run modal_app.py`)
# ---------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    import json
    import urllib.request

    url = fastapi_app.get_url()
    print(f"API base URL: {url}")

    health_url = f"{url}/health"
    print(f"Checking {health_url} …")
    with urllib.request.urlopen(health_url) as resp:
        print(json.loads(resp.read()))

