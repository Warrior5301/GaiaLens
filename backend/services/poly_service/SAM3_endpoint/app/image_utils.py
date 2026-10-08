import base64
import binascii
import io
import re

import cv2
import numpy as np
from fastapi import HTTPException
from PIL import Image

from app.config import settings

DATA_URL_RE = re.compile(r"^data:image/[a-zA-Z0-9.+-]+;base64,")


def decode_image(encoded: str) -> Image.Image:
    """Base64 string -> RGB PIL image. Raises HTTP errors for bad input."""
    encoded = DATA_URL_RE.sub("", encoded.strip(), count=1)
    encoded = re.sub(r"\s+", "", encoded)
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(status_code=400, detail="image must be valid base64")
    if not raw:
        raise HTTPException(status_code=400, detail="image is empty")
    if len(raw) > settings.MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="image too large")
    try:
        with Image.open(io.BytesIO(raw)) as im:
            if im.width * im.height > settings.MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="image dimensions too large")
            im.load()
            return im.convert("RGB")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="not a supported image")


def encode_image(image: Image.Image) -> str:
    """PIL image -> base64 PNG string."""
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_polygons(mask: np.ndarray, simplify: float) -> list[np.ndarray]:
    """Binary mask -> list of polygons (each an Nx2 array of x, y)."""
    contours, _ = cv2.findContours(
        mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    for c in contours:
        if cv2.contourArea(c) < settings.MIN_POLYGON_AREA:
            continue
        if simplify > 0:
            c = cv2.approxPolyDP(c, simplify, True)
        if len(c) >= 3:
            polys.append(c.reshape(-1, 2))
    return polys


def draw_overlay(
    image: Image.Image,
    masks: list[np.ndarray],
    polys_per_mask: list[list[np.ndarray]],
) -> Image.Image:
    """Draw colored masks and outlines on a copy of the image."""
    canvas = np.array(image).astype(np.float32)
    rng = np.random.default_rng(0)
    for mask, polys in zip(masks, polys_per_mask):
        color = rng.integers(60, 255, size=3).astype(np.float32)
        m = mask.astype(bool)
        canvas[m] = canvas[m] * 0.5 + color * 0.5
        out = canvas.astype(np.uint8)
        cv2.polylines(out, [p.astype(np.int32) for p in polys], True, color.tolist(), 2)
        canvas = out.astype(np.float32)
    return Image.fromarray(canvas.astype(np.uint8))
