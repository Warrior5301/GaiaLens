"""Compatibility layer exposing VisionRuntime, DeviceManager, and ModelCache."""

from app.core.device import DeviceManager
from app.core.model_cache import ModelCache
from app.core.vision_runtime import VisionRuntime

__all__ = ["VisionRuntime", "DeviceManager", "ModelCache"]
