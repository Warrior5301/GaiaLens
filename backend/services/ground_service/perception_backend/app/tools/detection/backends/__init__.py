"""Detection backends package for real remote-sensing oriented object detectors."""

from app.tools.detection.backends.rotated_yoloms import RotatedYOLOMSBackend
from app.tools.detection.backends.o2_rtdetr import O2RTDETRBackend

__all__ = ["RotatedYOLOMSBackend", "O2RTDETRBackend"]
