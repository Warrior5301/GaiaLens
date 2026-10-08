"""Safe image loading and format normalization service."""

from __future__ import annotations

from io import BytesIO
import os
from pathlib import Path
from urllib.parse import unquote, urlparse
from urllib.request import urlopen
import numpy as np
from PIL import Image
from app.core.logging import logger
from app.schemas.common import AssetRef

MAX_IMAGE_BYTES = int(os.getenv("MAX_IMAGE_BYTES", str(25 * 1024 * 1024)))


class ImageLoader:
    """Loads, verifies, and normalizes remote-sensing imagery referenced by AssetRef."""

    @staticmethod
    def load_pil(asset: AssetRef) -> Image.Image:
        """Load a local or HTTP(S) image as a PIL image in RGB format."""
        uri = asset.uri
        try:
            if uri.startswith("data:") and ";base64," in uri:
                import base64
                _, b64_data = uri.split(";base64,", 1)
                source = BytesIO(base64.b64decode(b64_data))
            else:
                parsed = urlparse(uri)
                if parsed.scheme in {"http", "https"}:
                    with urlopen(uri, timeout=45) as response:
                        contents = response.read(MAX_IMAGE_BYTES + 1)
                    if len(contents) > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"Asset exceeds MAX_IMAGE_BYTES ({MAX_IMAGE_BYTES} bytes): {asset.uri}"
                        )
                    source = BytesIO(contents)
                elif parsed.scheme in {"", "file"} or (
                    len(uri) >= 3 and uri[1] == ":" and uri[2] in {"\\", "/"}
                ):
                    local_path = (
                        unquote(parsed.path) if parsed.scheme == "file" else uri
                    )
                    if parsed.scheme == "file" and os.name == "nt" and local_path.startswith("/"):
                        local_path = local_path[1:]
                    path = Path(local_path)
                    if not path.exists():
                        raise FileNotFoundError(f"Asset file not found at: {asset.uri}")
                    if path.stat().st_size > MAX_IMAGE_BYTES:
                        raise ValueError(
                            f"Asset exceeds MAX_IMAGE_BYTES ({MAX_IMAGE_BYTES} bytes): {asset.uri}"
                        )
                    source = path
                else:
                    raise ValueError(
                        "Asset URI must be a local path, file:// URI, or HTTP(S) URL; "
                        f"got scheme '{parsed.scheme}'"
                    )

            with Image.open(source) as opened:
                return opened.convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load image from '{asset.uri}': {e}")
            raise

    @staticmethod
    def load_numpy(asset: AssetRef) -> np.ndarray:
        """Load image as uint8 NumPy array in HxWxC RGB order."""
        pil_img = ImageLoader.load_pil(asset)
        return np.array(pil_img, dtype=np.uint8)

    @staticmethod
    def verify_asset(asset: AssetRef) -> tuple[int, int]:
        """Verify image on disk and confirm dimension consistency.

        Returns:
            Tuple of (width_px, height_px).
        """
        img = ImageLoader.load_pil(asset)
        w, h = img.size
        if asset.width_px > 0 and asset.width_px != w:
            logger.warning(
                f"Asset width mismatch for {asset.asset_id}: declared {asset.width_px}, actual {w}"
            )
        if asset.height_px > 0 and asset.height_px != h:
            logger.warning(
                f"Asset height mismatch for {asset.asset_id}: declared {asset.height_px}, actual {h}"
            )
        return w, h
