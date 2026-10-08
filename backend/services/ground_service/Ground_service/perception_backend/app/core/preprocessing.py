"""Image preprocessing, tiling, crop extraction, and normalization utilities."""

from __future__ import annotations

from typing import Generator
import numpy as np
from PIL import Image
from app.schemas.common import BBoxPx, ROI


class ImagePreprocessor:
    """Provides standard remote-sensing preprocessing operations."""

    @staticmethod
    def resize(
        image: Image.Image,
        target_size: tuple[int, int] = (512, 512),
        resample: Image.Resampling = Image.Resampling.BILINEAR,
    ) -> Image.Image:
        """Resize PIL Image to target_size (width, height)."""
        return image.resize(target_size, resample=resample)

    @staticmethod
    def crop_roi(
        image: Image.Image,
        roi: ROI | BBoxPx,
    ) -> tuple[Image.Image, BBoxPx]:
        """Crop visual sub-region corresponding to ROI or BBoxPx."""
        box = roi.bbox_px if isinstance(roi, ROI) else roi
        w, h = image.size

        x1 = max(0, min(int(round(box.x_min)), w))
        y1 = max(0, min(int(round(box.y_min)), h))
        x2 = max(x1, min(int(round(box.x_max)), w))
        y2 = max(y1, min(int(round(box.y_max)), h))

        cropped = image.crop((x1, y1, x2, y2))
        clamped_box = BBoxPx(
            x_min=float(x1),
            y_min=float(y1),
            x_max=float(x2),
            y_max=float(y2),
        )
        return cropped, clamped_box

    @staticmethod
    def generate_sliding_tiles(
        image_shape: tuple[int, int],
        tile_size: int = 512,
        overlap: int = 64,
    ) -> Generator[BBoxPx, None, None]:
        """Generate sliding window bounding boxes covering an image of (height, width).

        Yields:
            BBoxPx for each tile in global pixel coordinates.
        """
        height, width = image_shape
        stride = tile_size - overlap
        if stride <= 0:
            raise ValueError(f"Tile size ({tile_size}) must be greater than overlap ({overlap})")

        y = 0
        while y < height:
            y_end = min(y + tile_size, height)
            y_start = max(0, y_end - tile_size)

            x = 0
            while x < width:
                x_end = min(x + tile_size, width)
                x_start = max(0, x_end - tile_size)

                yield BBoxPx(
                    x_min=float(x_start),
                    y_min=float(y_start),
                    x_max=float(x_end),
                    y_max=float(y_end),
                )

                if x_end >= width:
                    break
                x += stride

            if y_end >= height:
                break
            y += stride

    @staticmethod
    def normalize_to_tensor_array(
        image_array: np.ndarray,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> np.ndarray:
        """Normalize HxWxC uint8 image array to CxHxW float32 normalized array."""
        norm = image_array.astype(np.float32) / 255.0
        norm = (norm - np.array(mean, dtype=np.float32)) / np.array(std, dtype=np.float32)
        return np.transpose(norm, (2, 0, 1))
