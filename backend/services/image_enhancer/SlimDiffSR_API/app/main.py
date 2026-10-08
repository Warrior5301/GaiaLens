import os
import time
import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app.sr_service import SlimDiffSRService


# ---------------------------------------------------------
# Logging
# ---------------------------------------------------------

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

logger = logging.getLogger("slimdiffsr-api")


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_PATH = os.getenv(
    "SLIMDIFFSR_MODEL_PATH",
    "weight/net_params_4x.pkl",
)

SD_PATH = os.getenv(
    "SLIMDIFFSR_SD_PATH",
    "stabilityai/sd-turbo",
)

TAESD_PATH = os.getenv(
    "SLIMDIFFSR_TAESD_PATH",
    "madebyollin/taesd",
)

UPSCALE = int(
    os.getenv("SLIMDIFFSR_UPSCALE", "4")
)

PRUNE_DEPTH = int(
    os.getenv("SLIMDIFFSR_PRUNE_DEPTH", "2")
)

FP16 = os.getenv(
    "SLIMDIFFSR_FP16",
    "false",
).lower() in {"1", "true", "yes"}


# Maximum number of bytes accepted by the upload endpoint.
MAX_UPLOAD_BYTES = int(
    os.getenv(
        "SLIMDIFFSR_MAX_UPLOAD_BYTES",
        str(100 * 1024 * 1024),
    )
)


# ---------------------------------------------------------
# FastAPI
# ---------------------------------------------------------

app = FastAPI(
    title="SlimDiffSR API",
    version="1.1.0",
    description=(
        "Production-oriented API wrapper around the "
        "official SlimDiffSR inference pipeline."
    ),
)


service = SlimDiffSRService(
    model_path=MODEL_PATH,
    sd_path=SD_PATH,
    taesd_path=TAESD_PATH,
    upscale=UPSCALE,
    prune_depth=PRUNE_DEPTH,
    fp16=FP16,
)


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------

class ImageRequest(BaseModel):
    image_base64: str = Field(
        ...,
        description="Base64-encoded input image.",
    )



class ImageResponse(BaseModel):
    image_base64: str
    


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
    upscale: int


# ---------------------------------------------------------
# Health / readiness
# ---------------------------------------------------------

@app.get(
    "/health",
    response_model=HealthResponse,
)
def health():
    """
    Liveness endpoint.

    Tells us whether the API process is alive.
    """

    return HealthResponse(
        status="ok",
        model_loaded=service.is_loaded,
        device=service.device,
        upscale=service.upscale,
    )


@app.get("/ready")
def ready():
    """
    Readiness endpoint.

    The API is ready for inference only when the model
    has been loaded into memory.
    """

    if not service.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="SlimDiffSR model is not loaded.",
        )

    return {
        "status": "ready",
        "model_loaded": True,
        "device": service.device,
    }


# ---------------------------------------------------------
# Model lifecycle
# ---------------------------------------------------------

@app.post("/load")
def load_model():
    """
    Load SlimDiffSR into GPU memory.
    """

    try:
        start = time.perf_counter()

        service.load()

        elapsed = time.perf_counter() - start

        logger.info(
            "Model loaded in %.3f seconds",
            elapsed,
        )

        return {
            "status": "loaded",
            "model_loaded": service.is_loaded,
            "device": service.device,
            "load_time_seconds": round(elapsed, 3),
        }

    except Exception as exc:
        logger.exception("Model loading failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


@app.post("/unload")
def unload_model():
    """
    Unload SlimDiffSR and release GPU memory.
    """

    try:
        service.unload()

        logger.info("Model unloaded")

        return {
            "status": "unloaded",
            "model_loaded": service.is_loaded,
        }

    except Exception as exc:
        logger.exception("Model unloading failed")

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ---------------------------------------------------------
# Base64 inference endpoint
# ---------------------------------------------------------

@app.post(
    "/run",
    response_model=ImageResponse,
)
def super_resolve_base64(
    request: ImageRequest,
):
    """
    Base64 API.

    Useful for programmatic/agent-to-agent communication.
    """

    try:
        start = time.perf_counter()

        result = service.super_resolve(
            request.image_base64
        )

        elapsed = time.perf_counter() - start

        logger.info(
            "Base64 inference completed: "
            "%dx%d -> %dx%d in %.3f seconds",
            result["input_width"],
            result["input_height"],
            result["width"],
            result["height"],
            elapsed,
        )

        return ImageResponse(
            image_base64=result["image_base64"]
        )

    except RuntimeError as exc:
        raise HTTPException(
            status_code=503,
            detail=str(exc),
        )

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        )

    except Exception as exc:
        logger.exception(
            "Base64 inference failed"
        )

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )
