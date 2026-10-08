"""Unified VisionRuntime coordinating image loading, preprocessing, model caching, and hardware devices."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Generator, Optional
import numpy as np
from PIL import Image
from app.core.device import DeviceManager, DeviceType
from app.core.image_loader import ImageLoader
from app.core.logging import logger
from app.core.model_cache import ModelCache
from app.core.preprocessing import ImagePreprocessor
from app.schemas.common import AssetRef, BBoxPx, ROI


class VisionRuntime:
    """Central perception execution runtime for SAC GeoAI Agent tools."""

    def __init__(
        self,
        device_override: Optional[str] = None,
        model_cache: Optional[ModelCache] = None,
    ) -> None:
        self.device: DeviceType = DeviceManager.get_preferred_device(device_override)
        self.model_cache = model_cache or ModelCache()
        self.loader = ImageLoader()
        self.preprocessor = ImagePreprocessor()
        logger.info(f"Initialized VisionRuntime on device: {self.device}")

    def load_image(self, asset: AssetRef) -> Image.Image:
        """Load and verify image from asset reference as RGB PIL Image."""
        return self.loader.load_pil(asset)

    def load_image_array(self, asset: AssetRef) -> np.ndarray:
        """Load image as uint8 NumPy array in HxWxC RGB order."""
        return self.loader.load_numpy(asset)

    def crop_roi(self, image: Image.Image, roi: ROI | BBoxPx) -> tuple[Image.Image, BBoxPx]:
        """Crop visual sub-region based on ROI or BBoxPx."""
        return self.preprocessor.crop_roi(image, roi)

    def resize_for_model(
        self,
        image: Image.Image,
        target_size: tuple[int, int] = (512, 512),
    ) -> Image.Image:
        """Standardize image dimensions for model input."""
        return self.preprocessor.resize(image, target_size=target_size)

    def generate_tiles(
        self,
        image_shape: tuple[int, int],
        tile_size: int = 512,
        overlap: int = 64,
    ) -> Generator[BBoxPx, None, None]:
        """Generate sliding window bounding boxes covering an image of (height, width)."""
        return self.preprocessor.generate_sliding_tiles(
            image_shape=image_shape,
            tile_size=tile_size,
            overlap=overlap,
        )
