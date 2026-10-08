"""Asset management and inspection service for SAC GeoAI Agent.

Handles:
- Creation and validation of AssetRef from local or remote images
- Extraction of dimensions, bands, and geospatial metadata
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
from PIL import Image
from app.schemas.common import AssetRef


class AssetManager:
    """Manages ingestion, resolution, and caching of raw input assets."""

    @staticmethod
    def create_from_file(
        file_path: str | Path,
        asset_id: Optional[str] = None,
        modality: str = "optical",
        gsd_m: Optional[float] = None,
        crs: Optional[str] = None,
    ) -> AssetRef:
        """Create an AssetRef by inspecting an image file on disk."""
        path = Path(file_path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image asset file not found at: {path}")

        with Image.open(path) as img:
            width, height = img.size
            bands = list(img.getbands())

        aid = asset_id or path.stem

        return AssetRef(
            asset_id=aid,
            uri=str(path.as_posix()),
            modality=modality,
            width_px=width,
            height_px=height,
            gsd_m=gsd_m,
            crs=crs,
            bands=bands,
        )
