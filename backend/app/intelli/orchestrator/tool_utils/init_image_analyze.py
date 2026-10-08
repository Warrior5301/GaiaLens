import numpy as np
import requests
import base64
from io import BytesIO
from PIL import Image
from backend.app.intelli.common import GSD_URL


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert a base64-encoded string back to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))

def analyze_image(image: Image.Image) -> dict:
    """
    Lightweight image analysis for VLM/agent context.
    Reports only relatively reliable facts.
    """

    arr = np.asarray(image)
    width, height = image.size
    bands = list(image.getbands())

    # Check metadata for GSD
    gsd = None
    gsd_available = False

    for key in ("gsd", "GSD", "pixel_size", "resolution"):
        if key in image.info:
            gsd = image.info[key]
            gsd_available = True
            break
    if not gsd_available:
        # GSD service is optional; report unavailable instead of failing the analysis
        try:
            response = requests.post(
                f"{GSD_URL}/run",
                json={"image_base64": image_to_base64(image)},
                timeout=300
            )
            response.raise_for_status()
            gsd = response.json().get("gsd_m_per_pixel")
        except Exception:
            gsd = None
        gsd_available = gsd is not None

    result = {
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 3),
        "mode": image.mode,
        "num_bands": len(bands),
        "bands": bands,
        "dtype": str(arr.dtype),

        "gsd_available": gsd_available,
        "gsd": gsd,
    }

    # Simple, conservative image type classification
    if image.mode in ("RGB", "RGBA"):
        result["image_type"] = "visible_rgb"

    elif len(bands) == 1:
        result["image_type"] = "single_band"

    elif len(bands) > 3:
        result["image_type"] = "multiband"

    else:
        result["image_type"] = "unknown"

    return result