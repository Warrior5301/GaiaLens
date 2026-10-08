import base64
import binascii
import io
import re

import numpy as np
from fastapi import HTTPException
from PIL import Image, UnidentifiedImageError

from app.config import MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS

_DATA_URI = re.compile(r"^data:image/[A-Za-z0-9.+-]+;base64,")


def decode_image(encoded: str) -> Image.Image:
    """Decode a base64 string or image data URI into an RGB PIL image."""
    value = _DATA_URI.sub("", encoded.strip(), count=1)
    value = re.sub(r"\s+", "", value)
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be valid base64") from exc

    if not raw:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image file is too large")

    try:
        with Image.open(io.BytesIO(raw)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="image dimensions are too large")
            source.load()
            return source.convert("RGB")
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=400, detail="image must be a supported JPEG or PNG") from exc


def encode_png(image_rgb: np.ndarray) -> str:
    """Encode an RGB array as a base64 PNG string."""
    buffer = io.BytesIO()
    Image.fromarray(image_rgb.astype(np.uint8), mode="RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
