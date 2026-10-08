"""FastAPI inference service for sy1998/EarthMind-4B."""

from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, model_validator

logger = logging.getLogger("earthmind.api")
MODEL_ID = os.getenv("EARTHMIND_MODEL_ID", "sy1998/EarthMind-4B")
MODEL_REVISION = os.getenv(
    "EARTHMIND_MODEL_REVISION", "f17c66b01e540268f3862a540e84a3f7ee329cd3"
)
CACHE_DIR = Path(os.getenv("EARTHMIND_CACHE_DIR", "/root/.cache/earthmind")).expanduser()
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_FRAMES = 16

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

app = FastAPI(
    title="EarthMind-4B API",
    version="2.0.0",
    description=(
        "EarthMind image, temporal, multispectral, and masked-prompt inference. "
        "Images are sent as base64 and segmentation masks are returned as base64 PNGs."
    ),
)

_model: Any = None
_tokenizer: Any = None
_inference_lock = threading.Lock()


class InferenceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query_type: Literal[
        "normal", "temporal_analysis", "multispectral_analysis", "masked_prompt"
    ] = "normal"
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    past_text: str = ""
    image_base64: str | None = None
    images_base64: list[str] | None = None
    rgb_image_base64: str | None = None
    # Binary mask input: [region][row][column].
    mask_prompts: list[list[list[float]]] | None = None

    @model_validator(mode="after")
    def validate_query_inputs(self) -> "InferenceRequest":
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank")

        if self.query_type == "normal":
            if self.images_base64 or self.rgb_image_base64 or self.mask_prompts:
                raise ValueError("normal accepts image_base64 only; remove other image fields")
            if self.image_base64 and self.past_text:
                raise ValueError("past_text is only supported for text-only normal requests")

        elif self.query_type == "temporal_analysis":
            if not self.images_base64 or len(self.images_base64) < 2:
                raise ValueError("temporal_analysis requires at least two images_base64 frames")
            if len(self.images_base64) > MAX_VIDEO_FRAMES:
                raise ValueError(f"temporal_analysis accepts at most {MAX_VIDEO_FRAMES} frames")
            if self.image_base64 or self.rgb_image_base64 or self.mask_prompts or self.past_text:
                raise ValueError("temporal_analysis accepts images_base64 and prompt only")

        elif self.query_type == "multispectral_analysis":
            if not self.image_base64 or not self.rgb_image_base64:
                raise ValueError(
                    "multispectral_analysis requires image_base64 and rgb_image_base64"
                )
            if self.images_base64 or self.mask_prompts or self.past_text:
                raise ValueError(
                    "multispectral_analysis does not accept images_base64, masks, or past_text"
                )

        elif self.query_type == "masked_prompt":
            if not self.image_base64 or not self.mask_prompts:
                raise ValueError("masked_prompt requires image_base64 and mask_prompts")
            if self.images_base64 or self.rgb_image_base64 or self.past_text:
                raise ValueError("masked_prompt accepts image_base64 and mask_prompts only")
            height = len(self.mask_prompts[0]) if self.mask_prompts else 0
            width = len(self.mask_prompts[0][0]) if height else 0
            if not height or not width or any(
                len(region) != height or any(len(row) != width for row in region)
                for region in self.mask_prompts
            ):
                raise ValueError("Every mask must be a non-empty rectangular 2D array of equal size")
            if any(
                value not in (0, 1, 0.0, 1.0)
                for region in self.mask_prompts
                for row in region
                for value in row
            ):
                raise ValueError("mask_prompts pixels must be 0 or 1")
        return self


def _patch_remote_config(model_path: str) -> None:
    """Patch the actual config class named by this snapshot's auto_map.

    Transformers 4.42.3 formats a loaded config by instantiating its class with
    no arguments. EarthMind's Sa2VAChatConfig constructor expects llm_config to
    contain architectures, so that formatting call raises KeyError. Serializing
    the loaded config directly avoids constructing an invalid empty default.
    """
    config_path = Path(model_path) / "config.json"
    config_data = json.loads(config_path.read_text(encoding="utf-8"))
    config_ref = config_data.get("auto_map", {}).get("AutoConfig")
    if isinstance(config_ref, list):
        config_ref = config_ref[0]
    if not isinstance(config_ref, str) or "." not in config_ref:
        raise RuntimeError(f"Model config has no valid AutoConfig mapping: {config_ref!r}")

    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    config_class = get_class_from_dynamic_module(config_ref, model_path)
    config_class.to_diff_dict = lambda self: self.to_dict()


def _load_model() -> tuple[Any, Any]:
    """Load EarthMind once in this server process."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("EarthMind-4B requires a CUDA GPU")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("EarthMind-4B requires a CUDA GPU with BF16 support")

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(CACHE_DIR),
    )
    _patch_remote_config(model_path)

    # Match the model publisher's documented AutoModelForCausalLM loading path.
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    model.preparing_for_generation(
        tokenizer=tokenizer,
        max_new_tokens=128,
        torch_dtype=torch.bfloat16,
    )
    _model, _tokenizer = model, tokenizer
    logger.info("Loaded %s at revision %s", MODEL_ID, MODEL_REVISION)
    return _model, _tokenizer


def _decode_image(encoded: str, field_name: str) -> Image.Image:
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError(f"{field_name} must be raw base64 or a base64 data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} is not valid base64") from exc
    if not raw:
        raise ValueError(f"{field_name} is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"{field_name} exceeds the 12 MiB image limit")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{field_name} is not a supported image") from exc


def _encode_mask(mask: Any) -> dict[str, Any]:
    array = np.asarray(mask)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("Model returned a mask with an unsupported shape")
    if array.dtype == np.bool_ or (array.size and np.nanmax(array) <= 1):
        array = array.astype(np.uint8) * 255
    else:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array)
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return {
        "encoding": "base64",
        "media_type": "image/png",
        "width": image.width,
        "height": image.height,
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def _encode_masks(raw_masks: Any) -> list[list[dict[str, Any]]]:
    """Return masks grouped as [segmentation][frame], each encoded as a base64 PNG."""
    if raw_masks is None:
        return []
    encoded: list[list[dict[str, Any]]] = []
    for group in raw_masks:
        array = np.asarray(group)
        if array.ndim == 2:
            frames = [array]
        elif array.ndim == 3:
            frames = list(array)
        else:
            raise ValueError("Model returned masks with an unsupported shape")
        encoded.append([_encode_mask(frame) for frame in frames])
    return encoded


def _infer(request: InferenceRequest) -> dict[str, Any]:
    prompt = request.prompt.strip()
    text = f"<image>\n{prompt}"

    if request.query_type == "normal":
        image = _decode_image(request.image_base64, "image_base64") if request.image_base64 else None
        args = {
            "image": image,
            "text": text if image is not None else prompt,
            "past_text": request.past_text,
        }
        method = "predict_forward"
    elif request.query_type == "temporal_analysis":
        frames = [
            _decode_image(value, f"images_base64[{index}]")
            for index, value in enumerate(request.images_base64 or [])
        ]
        args = {"video": frames, "text": text}
        method = "predict_forward"
    elif request.query_type == "multispectral_analysis":
        args = {
            "image": _decode_image(request.image_base64 or "", "image_base64"),
            "rgb_image": _decode_image(request.rgb_image_base64 or "", "rgb_image_base64"),
            "text": text,
        }
        method = "predict_forward_multi"
    else:
        args = {
            "image": _decode_image(request.image_base64 or "", "image_base64"),
            "text": text,
            "mask_prompts": [request.mask_prompts],
        }
        method = "predict_forward"

    with _inference_lock, torch.inference_mode():
        model, tokenizer = _load_model()
        if not hasattr(model, "gen_config"):
            raise RuntimeError("Loaded EarthMind model has no gen_config")
        model.gen_config.max_new_tokens = request.max_new_tokens
        args["tokenizer"] = tokenizer
        started = time.perf_counter()
        result = getattr(model, method)(**args)

    return {
        "query_type": request.query_type,
        "prediction": str(result.get("prediction", ""))
        .replace("<|im_end|>", "")
        .strip(),
        "prediction_masks": _encode_masks(result.get("prediction_masks", [])),
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }


@app.get("/")
def root() -> dict[str, str]:
    return {"name": "EarthMind-4B API", "health": "/health", "inference": "POST /run"}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REVISION}


@app.post("/load")
def load() -> dict[str, str]:
    """Optionally warm the model before sending inference requests."""
    try:
        with _inference_lock:
            _load_model()
        return {"status": "loaded", "model": MODEL_ID}
    except Exception as exc:
        logger.exception("EarthMind model load failed")
        raise HTTPException(
            status_code=500, detail=f"Model load failed: {type(exc).__name__}: {exc}"
        ) from exc


@app.post("/run")
def infer(request: InferenceRequest) -> dict[str, Any]:
    try:
        return _infer(request)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("EarthMind inference failed")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}: {exc}",
        ) from exc
