"""Modal deployment entrypoint for GaiaLens Moondream 3."""

import modal


MODEL_ID = "moondream/moondream3-preview"
MODEL_REVISION = "5112966d1a723413b1c9a1e8bea272b72e647b35"
CACHE_DIR = "/cache/huggingface"
MODULES_CACHE_DIR = "/tmp/moondream-transformers-modules"

app = modal.App("gaialens-moondream")
model_cache = modal.Volume.from_name("gaialens-moondream-cache", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "fastapi==0.115.12",
        "pydantic==2.11.3",
        "Pillow>=11.0.0",
        "numpy<2",
        "torch==2.7.1",
        "torchvision==0.22.1",
        "transformers==4.51.3",
        "tokenizers==0.21.1",
        "huggingface_hub[hf_xet]>=0.23.2,<1.0",
        "accelerate==1.10.1",
        "safetensors",
        "einops",
        "timm",
    )
    .env(
        {
            "MOONDREAM_MODEL_ID": MODEL_ID,
            "MOONDREAM_MODEL_REVISION": MODEL_REVISION,
            "MOONDREAM_CACHE_DIR": CACHE_DIR,
            "HF_HOME": CACHE_DIR,
            "HF_HUB_CACHE": f"{CACHE_DIR}/hub",
            "HF_MODULES_CACHE": MODULES_CACHE_DIR,
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    .add_local_file("app.py", remote_path="/root/app.py")
)


@app.function(
    image=image,
    # Use a GPU fallback so Modal can schedule the app on either pool.
    gpu=["A100-40GB", "L40S"],
    timeout=900,
    scaledown_window=300,
    volumes={"/cache": model_cache},
)
@modal.asgi_app()
def fastapi_app():
    from app import app as moondream_app

    return moondream_app
