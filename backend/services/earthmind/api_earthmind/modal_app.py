"""Modal deployment entrypoint for the EarthMind-4B FastAPI service.

Run from this directory with modal serve modal_app.py for development or
modal deploy modal_app.py for a persistent deployment.
"""

import modal


MODEL_ID = "sy1998/EarthMind-4B"
MODEL_REVISION = "f17c66b01e540268f3862a540e84a3f7ee329cd3"
CACHE_DIR = "/root/.cache/earthmind"

app = modal.App("earthmind-api")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.12",
        "pydantic==2.11.3",
        "Pillow",
        "numpy<2",
        "torch==2.5.1",
        "torchvision==0.20.1",
        "transformers==4.42.3",
        "huggingface_hub[hf_xet]>=0.23.2,<1.0",
        "accelerate",
        "einops",
        "timm",
        "peft==0.11.1",
        "sentencepiece",
        "safetensors",
        "tqdm",
    )
    .env(
        {
            "EARTHMIND_MODEL_ID": MODEL_ID,
            "EARTHMIND_MODEL_REVISION": MODEL_REVISION,
            "EARTHMIND_CACHE_DIR": CACHE_DIR,
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "TRANSFORMERS_CACHE": f"{CACHE_DIR}/hub",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    # Bake the pinned model snapshot into the image so inference does not need
    # to download weights during a web request or container startup.
    .run_commands(
        "python -c 'from huggingface_hub import snapshot_download; "
        "snapshot_download("
        "repo_id=\"sy1998/EarthMind-4B\", "
        "revision=\"f17c66b01e540268f3862a540e84a3f7ee329cd3\", "
        "cache_dir=\"/root/.cache/earthmind\""
        ")'"
    )
    .add_local_file("app.py", remote_path="/root/app.py")
)


@app.function(
    image=image,
    gpu="A100-40GB",
    timeout=900,
    scaledown_window=300,
)
@modal.asgi_app()
def fastapi_app():
    from app import app as earthmind_app

    return earthmind_app

