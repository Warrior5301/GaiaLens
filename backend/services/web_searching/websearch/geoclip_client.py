"""Image decoding and lazy GeoCLIP inference for the location-context API."""

from __future__ import annotations

import base64
import binascii
import logging
import os
import tempfile
import threading
from io import BytesIO
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)
MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(10 * 1024 * 1024)))
MAX_IMAGE_PIXELS = int(os.getenv("MAX_IMAGE_PIXELS", "40000000"))
MAX_IMAGE_BASE64_CHARS = 4 * ((MAX_IMAGE_BYTES + 2) // 3) + 1024


class InvalidImageError(ValueError):
    """Raised when an uploaded Base64 value is not a supported image."""


class GeoCLIPUnavailableError(RuntimeError):
    """Raised when GeoCLIP or its pretrained model weights cannot be loaded."""


def decode_image_base64(value: str) -> bytes:
    """Decode raw Base64 or a Base64 image data URI with size checks."""
    payload = value.strip()
    if payload.lower().startswith("data:"):
        header, separator, payload = payload.partition(",")
        if not separator or not header.lower().startswith("data:image/") or ";base64" not in header.lower():
            raise InvalidImageError("image_base64 must be raw Base64 or a Base64 image data URI")

    # Reject oversized values before allocating the decoded buffer.
    max_encoded_size = 4 * ((MAX_IMAGE_BYTES + 2) // 3)
    if len(payload) > max_encoded_size:
        raise InvalidImageError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte upload limit")

    try:
        image_bytes = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidImageError("image_base64 is not valid Base64") from exc
    if not image_bytes:
        raise InvalidImageError("The uploaded image is empty")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise InvalidImageError(f"Image exceeds the {MAX_IMAGE_BYTES}-byte upload limit")
    return image_bytes


class GeoCLIPGeolocator:
    """Load GeoCLIP once and predict GPS candidates for uploaded images."""

    def __init__(self) -> None:
        self._model: Any | None = None
        self._torch: Any | None = None
        self._model_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    def _load_model(self) -> tuple[Any, Any]:
        if self._model is not None and self._torch is not None:
            return self._model, self._torch

        with self._model_lock:
            if self._model is not None and self._torch is not None:
                return self._model, self._torch
            try:
                import torch
                from geoclip import GeoCLIP

                model = GeoCLIP()
                device_name = os.getenv("GEOCLIP_DEVICE", "").strip()
                if device_name:
                    device = torch.device(device_name)
                else:
                    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model.to(device)
                model.eval()
            except Exception as exc:
                logger.exception("Could not initialize GeoCLIP")
                raise GeoCLIPUnavailableError(
                    "GeoCLIP could not be initialized. Install the GeoCLIP/PyTorch dependencies "
                    "and ensure its pretrained weights are available."
                ) from exc

            self._model = model
            self._torch = torch
            return model, torch

    @staticmethod
    def _validate_image(image_bytes: bytes) -> str:
        try:
            from PIL import Image, UnidentifiedImageError

            with Image.open(BytesIO(image_bytes)) as image:
                width, height = image.size
                if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                    raise InvalidImageError(
                        f"Image dimensions exceed the {MAX_IMAGE_PIXELS}-pixel limit"
                    )
                image_format = (image.format or "").lower()
                image.verify()
        except InvalidImageError:
            raise
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise InvalidImageError("The uploaded data is not a readable image") from exc

        suffixes = {"jpeg": ".jpg", "png": ".png", "webp": ".webp", "bmp": ".bmp", "gif": ".gif", "tiff": ".tiff"}
        return suffixes.get(image_format, ".img")

    def predict(self, image_bytes: bytes, top_k: int = 5) -> list[dict[str, float]]:
        """Return GeoCLIP's top GPS candidates and probabilities."""
        if top_k < 1:
            raise ValueError("top_k must be at least 1")
        suffix = self._validate_image(image_bytes)
        model, torch = self._load_model()

        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as image_file:
                image_file.write(image_bytes)
                temporary_path = image_file.name

            with self._inference_lock, torch.inference_mode():
                gps_candidates, probabilities = model.predict(temporary_path, top_k=top_k)

            predictions: list[dict[str, float]] = []
            for gps, probability in zip(gps_candidates, probabilities):
                predictions.append({
                    "latitude": float(gps[0]),
                    "longitude": float(gps[1]),
                    "probability": float(probability),
                })
            if not predictions:
                raise GeoCLIPUnavailableError("GeoCLIP returned no coordinate predictions")
            return predictions
        except GeoCLIPUnavailableError:
            raise
        except Exception as exc:
            logger.exception("GeoCLIP inference failed")
            raise GeoCLIPUnavailableError("GeoCLIP could not complete image geolocation") from exc
        finally:
            if temporary_path:
                try:
                    Path(temporary_path).unlink(missing_ok=True)
                except OSError:
                    logger.warning("Could not remove temporary GeoCLIP image: %s", temporary_path)
