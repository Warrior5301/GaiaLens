"""FastAPI service for PP-OCRv5 Mobile text recognition."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import re
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("ppocr_api")
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")


class OCRRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    image: str = Field(..., description="Base64-encoded image, optionally as a data URL")


class OCRResult(BaseModel):
    text: str
    lines: list[str]


def _decode_image(encoded: str) -> bytes:
    encoded = DATA_URL_RE.sub("", encoded.strip(), count=1)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must contain valid base64 data") from exc
    if not raw:
        raise HTTPException(status_code=400, detail="image must not be empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail=f"image exceeds {MAX_IMAGE_BYTES} byte limit")
    return raw


def _extract_lines(result: Any) -> list[str]:
    """Normalize PaddleOCR's predict() output to recognized text lines."""
    lines: list[str] = []
    for page in result or []:
        # PaddleOCR 3.x returns result objects exposing json; dict support keeps
        # this adapter compatible with serialized/injected results as well.
        payload = getattr(page, "json", page)
        if callable(payload):
            payload = payload()
        if isinstance(payload, dict) and isinstance(payload.get("res"), dict):
            payload = payload["res"]
        if isinstance(payload, dict):
            texts = payload.get("rec_texts", [])
            lines.extend(str(text) for text in texts if text)
            continue
        # Compatibility with the classic PaddleOCR nested tuple format.
        if isinstance(page, (list, tuple)):
            for item in page:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    recognized = item[1]
                    if isinstance(recognized, (list, tuple)) and recognized:
                        lines.append(str(recognized[0]))
    return lines


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.ocr = None
    app.state.ocr_error = None
    try:
        from paddleocr import PaddleOCR

        # PaddleOCR 3.x bundled PP-OCRv5 Mobile pipeline. Model files are
        # downloaded by PaddleOCR on first startup if not cached locally.
        app.state.ocr = PaddleOCR(
            text_detection_model_name="PP-OCRv5_mobile_det",
            text_recognition_model_name="PP-OCRv5_mobile_rec",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
    except Exception as exc:  # Keep health/docs endpoints available on setup failure.
        app.state.ocr_error = str(exc)
        logger.exception("Unable to initialize PaddleOCR")
    yield


app = FastAPI(title="PP-OCRv5 Mobile API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health() -> dict[str, str]:
    if app.state.ocr is None:
        raise HTTPException(status_code=503, detail="OCR model is not ready")
    return {"status": "ok", "model": "PP-OCRv5 Mobile"}


@app.post("/run", response_model=OCRResult)
def recognize(request: OCRRequest) -> OCRResult:
    import numpy as np
    from PIL import Image, UnidentifiedImageError
    from io import BytesIO

    if app.state.ocr is None:
        raise HTTPException(status_code=503, detail="OCR model is not ready; check server logs")
    raw = _decode_image(request.image)
    try:
        with Image.open(BytesIO(raw)) as image:
            image.load()
            rgb = image.convert("RGB")
            pixels = np.ascontiguousarray(np.asarray(rgb)[:, :, ::-1])  # PaddleOCR expects BGR arrays
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image data is not a supported image") from exc
    try:
        lines = _extract_lines(app.state.ocr.predict(pixels))
    except Exception as exc:
        logger.exception("OCR inference failed")
        raise HTTPException(status_code=500, detail="OCR inference failed") from exc
    return OCRResult(text="\n".join(lines), lines=lines)
