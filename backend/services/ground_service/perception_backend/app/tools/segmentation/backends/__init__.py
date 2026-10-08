"""Real remote-sensing segmentation backends."""

from app.tools.segmentation.backends.segearth_ov3 import SegEarthOV3Backend
from app.tools.segmentation.backends.sam3_real import SAM3RealBackend

__all__ = ["SegEarthOV3Backend", "SAM3RealBackend"]
