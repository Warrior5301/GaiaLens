"""Hardware device and accelerator detection for SAC GeoAI Agent."""

from __future__ import annotations

import os
from typing import Literal

DeviceType = Literal["cuda", "mps", "cpu"]


class DeviceManager:
    """Manages computation devices (CUDA, MPS, CPU)."""

    @staticmethod
    def get_preferred_device(override: str | None = None) -> DeviceType:
        """Determine optimal available hardware accelerator.

        Args:
            override: Optional manual device override ("cuda", "mps", "cpu", "auto").
        """
        if override and override.lower() not in ("auto", "none", ""):
            ov = override.lower()
            if ov in ("cuda", "mps", "cpu"):
                return ov  # type: ignore[return-value]

        # Check environment variable
        env_dev = os.getenv("SAC_GEOAI_RUNTIME_DEVICE", "auto").lower()
        if env_dev in ("cuda", "mps", "cpu"):
            return env_dev  # type: ignore[return-value]

        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"

    @staticmethod
    def get_device_info() -> dict[str, str | int | float]:
        """Return diagnostic dictionary of device state."""
        dev = DeviceManager.get_preferred_device()
        info: dict[str, str | int | float] = {"preferred_device": dev}

        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_device_count"] = torch.cuda.device_count()
                info["cuda_device_name"] = torch.cuda.get_device_name(0)
        except ImportError:
            info["torch_version"] = "not_installed"

        return info
