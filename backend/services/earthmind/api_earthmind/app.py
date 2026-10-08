"""Standard FastAPI inference service for EarthMind-4B."""

import base64
import binascii
import io
import os
import threading
import time
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field, model_validator
from PIL import Image, UnidentifiedImageError


MODEL_ID = os.getenv("EARTHMIND_MODEL_ID", "sy1998/EarthMind-4B")
MODEL_REVISION = os.getenv(
    "EARTHMIND_MODEL_REVISION",
    "f17c66b01e540268f3862a540e84a3f7ee329cd3",
)
CACHE_DIR = Path(
    os.getenv("EARTHMIND_CACHE_DIR", str(Path.home() / ".cache" / "earthmind"))
).expanduser()
MAX_IMAGE_BYTES = 12 * 1024 * 1024
MAX_VIDEO_FRAMES = 16

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

app = FastAPI(
    title="EarthMind-4B API",
    version="1.0.0",
    description="Image, temporal, multisensor, and masked-prompt inference for EarthMind-4B.",
)

_model: Any = None
_tokenizer: Any = None
_inference_lock = threading.Lock()


class InferenceRequest(BaseModel):
    """Request body for each supported EarthMind inference mode."""

    model_config = ConfigDict(extra="forbid")

    query_type: Literal[
        "normal",
        "temporal_analysis",
        "multispectral_analysis",
        "masked_prompt",
    ]
    prompt: str = Field(min_length=1, max_length=8000)
    max_new_tokens: int = Field(default=128, ge=1, le=512)
    past_text: str = ""

    # normal / masked_prompt use image_base64; temporal_analysis uses
    # images_base64 in chronological order; multispectral_analysis uses
    # image_base64 with its paired rgb_image_base64 input.
    image_base64: str | None = None
    images_base64: list[str] | None = None
    rgb_image_base64: str | None = None
    # Shape: [region][row][column], with binary values 0 or 1.
    mask_prompts: list[list[list[float]]] | None = None

    @model_validator(mode="after")
    def validate_query_inputs(self):
        if not self.prompt.strip():
            raise ValueError("prompt cannot be blank.")
        if self.query_type == "normal":
            if self.images_base64 or self.rgb_image_base64 or self.mask_prompts:
                raise ValueError("normal accepts at most one image_base64 image.")
            if self.image_base64 and self.past_text:
                raise ValueError("past_text can only be combined with a text-only normal request.")
        elif self.query_type == "temporal_analysis":
            if self.images_base64 is None or len(self.images_base64) < 2:
                raise ValueError("temporal_analysis requires at least two images_base64 frames.")
            if len(self.images_base64) > MAX_VIDEO_FRAMES:
                raise ValueError(f"temporal_analysis accepts at most {MAX_VIDEO_FRAMES} frames.")
            if self.image_base64 or self.rgb_image_base64 or self.mask_prompts:
                raise ValueError("temporal_analysis uses images_base64 only.")
            if self.past_text:
                raise ValueError("past_text is only supported for text-only normal requests.")
        elif self.query_type == "multispectral_analysis":
            if not self.image_base64 or not self.rgb_image_base64:
                raise ValueError("multispectral_analysis requires image_base64 and rgb_image_base64.")
            if self.images_base64 or self.mask_prompts:
                raise ValueError("multispectral_analysis does not accept images_base64 or mask_prompts.")
            if self.past_text:
                raise ValueError("past_text is only supported for text-only normal requests.")
        elif self.query_type == "masked_prompt":
            if not self.image_base64 or not self.mask_prompts:
                raise ValueError("masked_prompt requires image_base64 and mask_prompts.")
            if self.images_base64 or self.rgb_image_base64:
                raise ValueError("masked_prompt uses image_base64 and mask_prompts only.")
            if self.past_text:
                raise ValueError("past_text is only supported for text-only normal requests.")
            height = len(self.mask_prompts[0]) if self.mask_prompts else 0
            width = len(self.mask_prompts[0][0]) if height else 0
            if not height or not width or any(
                len(region) != height or any(len(row) != width for row in region)
                for region in self.mask_prompts
            ):
                raise ValueError("Each mask region must be a non-empty rectangular 2D array of equal size.")
            if any(
                value not in (0, 1, 0.0, 1.0)
                for region in self.mask_prompts
                for row in region
                for value in row
            ):
                raise ValueError("mask_prompts pixels must be 0 or 1.")
        return self


def _load_model() -> tuple[Any, Any]:
    """Load the model once in this server process."""
    global _model, _tokenizer
    if _model is not None and _tokenizer is not None:
        return _model, _tokenizer

    from huggingface_hub import snapshot_download
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError(
            "EarthMind-4B requires a CUDA GPU. Run this API on a CUDA-capable host."
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The CUDA GPU must support BF16 for this model configuration.")

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(CACHE_DIR),
    )
    _tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=False,
        local_files_only=True,
    )
    _model = AutoModel.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    _model.eval().cuda()
    _model.preparing_for_generation(
        tokenizer=_tokenizer,
        max_new_tokens=128,
        torch_dtype=torch.bfloat16,
    )
    return _model, _tokenizer


def _decode_image(encoded: str, *, field_name: str) -> Image.Image:
    # Accept raw base64 or a data URL such as data:image/png;base64,...
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError(f"{field_name} must be a base64 image or base64 data URL.")
    try:
        image_bytes = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError(f"{field_name} is not valid base64.") from exc
    if not image_bytes:
        raise ValueError(f"{field_name} is empty.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise ValueError(f"{field_name} exceeds the 12 MiB per-image limit.")
    try:
        with Image.open(io.BytesIO(image_bytes)) as opened:
            opened.load()
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"{field_name} is not a supported image.") from exc


def _encode_mask(mask: Any) -> dict[str, Any]:
    array = np.asarray(mask)
    while array.ndim > 2:
        array = array[0]
    if array.ndim != 2:
        raise ValueError("The model returned a mask with an unsupported shape.")
    if array.dtype == np.bool_ or (array.size and np.nanmax(array) <= 1):
        array = array.astype(np.uint8) * 255
    else:
        array = np.nan_to_num(array, nan=0.0, posinf=255.0, neginf=0.0)
        array = np.clip(array, 0, 255).astype(np.uint8)
    image = Image.fromarray(array, mode="L")
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
    """Encode returned masks as [segmentation][frame] base64 PNGs."""
    if raw_masks is None:
        return []
    encoded_groups: list[list[dict[str, Any]]] = []
    for group in raw_masks:
        array = np.asarray(group)
        if array.ndim == 2:
            frames = [array]
        elif array.ndim == 3:
            frames = list(array)
        else:
            raise ValueError("The model returned masks with an unsupported shape.")
        encoded_groups.append([_encode_mask(frame) for frame in frames])
    return encoded_groups


@app.get("/")
def root():
    return {"name": "EarthMind-4B API", "docs": "/docs", "health": "/health"}


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REVISION}


@app.post("/run")
def infer(request: InferenceRequest):
    try:
        text = f"<image>\n{request.prompt.strip()}"

        if request.query_type == "normal":
            image = (
                _decode_image(request.image_base64, field_name="image_base64")
                if request.image_base64
                else None
            )
            infer_args = {
                "image": image,
                "text": text if image is not None else request.prompt.strip(),
                "past_text": request.past_text,
            }
            call_name = "predict_forward"
        elif request.query_type == "temporal_analysis":
            frames = [
                _decode_image(encoded, field_name=f"images_base64[{index}]")
                for index, encoded in enumerate(request.images_base64 or [])
            ]
            infer_args = {"video": frames, "text": text}
            call_name = "predict_forward"
        elif request.query_type == "multispectral_analysis":
            infer_args = {
                "image": _decode_image(request.image_base64, field_name="image_base64"),
                "rgb_image": _decode_image(request.rgb_image_base64, field_name="rgb_image_base64"),
                "text": text,
            }
            call_name = "predict_forward_multi"
        else:
            infer_args = {
                "image": _decode_image(request.image_base64, field_name="image_base64"),
                "text": text,
                "mask_prompts": [request.mask_prompts],
            }
            call_name = "predict_forward"

        # The model mutates its generation configuration and uses CUDA; serialize
        # model loading and inference when this process receives concurrent calls.
        with _inference_lock:
            model, tokenizer = _load_model()
            model.gen_config.max_new_tokens = request.max_new_tokens
            infer_args["tokenizer"] = tokenizer
            call = getattr(model, call_name)

            started = time.perf_counter()
            with torch.inference_mode():
                result = call(**infer_args)

        return {
            "query_type": request.query_type,
            "prediction": str(result.get("prediction", "")).replace("<|im_end|>", "").strip(),
            "prediction_masks": _encode_masks(result.get("prediction_masks", [])),
            "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        print(f"EarthMind inference failed: {type(exc).__name__}: {exc}")
        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {type(exc).__name__}.",
        ) from exc
