"""
Qwen3-VL served by vLLM on Modal (OpenAI-compatible API, LoRA hot-loading enabled).

Deploy:   modal deploy backend/app/modal_vllm/vllm_server.py
Then run the backend with:   export QWEN_URL=https://<workspace>--<app>-server.modal.run/v1
(the exact URL is printed by `modal deploy`)

Optional deploy-time settings (environment variables):
  MIN_CONTAINERS=1     keep one GPU container always running (default 0: scales to zero when idle,
                       the first request after that waits for the model to load; an always-on GPU is billed)
  VLLM_API_KEY=...     make vLLM require this key. The backend does not send a key yet, so leave it unset for now.
"""
import json
import os
import subprocess
from pathlib import Path

import modal

MINUTES = 60
MODEL_NAME = "Qwen/Qwen3-VL-8B-Instruct"   # must match QWEN_MODEL in the backend
VLLM_PORT = 8000
GPU = "A100-40GB"

MAX_MODEL_LEN = 16384        # one agent request carries several images plus tool output
MAX_IMAGES_PER_PROMPT = 10
MAX_LORA_RANK = 64           # >= the highest rank among your adapters (allowed: 1, 8, 16, 32, 64, 128, 256, 320, 512)
MAX_LORAS = 4                # adapters usable in one batch (the backend has 4)
ENABLE_TOWER_CONNECTOR_LORA = True   # only needed if an adapter has vision-tower/connector weights
FAST_BOOT = True             # --enforce-eager: faster cold start, slower decoding

MIN_CONTAINERS = int(os.getenv("MIN_CONTAINERS", "0"))
API_KEY = os.getenv("VLLM_API_KEY", "")

LORA_DIR = Path(__file__).parent / "lora"

app = modal.App("gaialens-qwen-vllm")

hf_cache = modal.Volume.from_name("gaialens-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name("gaialens-vllm-cache", create_if_missing=True)

image = (
    modal.Image.from_registry("nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12")
    .entrypoint([])
    .uv_pip_install("vllm==0.21.0")   # the version Modal's vLLM example pins
    .env(
        {
            "HF_XET_HIGH_PERFORMANCE": "1",
            # lets the backend load adapters at runtime through POST /v1/load_lora_adapter
            "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "True",
        }
    )
    .add_local_dir(LORA_DIR, "/loras")   # keep last: copies the local lora/ folder into the container
)


@app.server(
    image=image,
    gpu=GPU,
    port=VLLM_PORT,
    startup_timeout=10 * MINUTES,
    scaledown_window=15 * MINUTES,
    min_containers=MIN_CONTAINERS,
    target_concurrency=16,
    volumes={
        "/root/.cache/huggingface": hf_cache,
        "/root/.cache/vllm": vllm_cache,
    },
    secrets=[modal.Secret.from_dict({"VLLM_API_KEY": API_KEY})] if API_KEY else [],
    # The backend cannot send Modal auth headers, so the URL is public. Runtime LoRA loading is enabled,
    # so set VLLM_API_KEY (and update the backend to send it) before sharing the URL.
    unauthenticated=True,
)
class Server:
    @modal.enter()
    def start(self):
        cmd = [
            "vllm", "serve", MODEL_NAME,
            "--host", "0.0.0.0",
            "--port", str(VLLM_PORT),
            "--dtype", "bfloat16",
            "--max-model-len", str(MAX_MODEL_LEN),
            "--limit-mm-per-prompt", json.dumps({"image": MAX_IMAGES_PER_PROMPT}),
            "--enable-lora",
            "--max-lora-rank", str(MAX_LORA_RANK),
            "--max-loras", str(MAX_LORAS),
            "--max-cpu-loras", str(MAX_LORAS),
            "--uvicorn-log-level=info",
            "--enforce-eager" if FAST_BOOT else "--no-enforce-eager",
        ]
        if ENABLE_TOWER_CONNECTOR_LORA:
            cmd.append("--enable-tower-connector-lora")
        self.process = subprocess.Popen(cmd)

    @modal.exit()
    def stop(self):
        self.process.terminate()
