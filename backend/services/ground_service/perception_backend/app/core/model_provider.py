"""Abstract Model Provider interface for hosting-agnostic model lifecycle.

Decouples the control and orchestration layer from specific hosting runtimes
(Modal, Local GPU, RunPod, Kubernetes, generic HTTP).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ModelProvider(ABC):
    """Abstract interface defining the lifecycle operations for any model hosting backend."""

    def __init__(self, provider_name: str, config: Optional[dict[str, Any]] = None) -> None:
        self.provider_name = provider_name
        self.config = config or {}

    @abstractmethod
    def load(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Request the provider to load/initialize the specified model.

        Args:
            model_config: Dictionary containing model_id, checkpoint, precision,
                          device, endpoints, etc.

        Returns:
            Dictionary containing load status, model_instance_id, provider,
            real_model flag, device, dtype, checkpoint, and load_latency_ms.
        """
        pass

    @abstractmethod
    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute model forward pass inference on prepared inputs.

        Args:
            request: Standardized inference request dictionary (image_base64,
                     query/target_classes, parameters, etc.).

        Returns:
            Standardized inference response dictionary containing detections/groundings,
            model metadata, and processing metrics.
        """
        pass

    @abstractmethod
    def unload(self) -> dict[str, Any]:
        """Request the provider to release/unload the currently loaded model.

        Returns:
            Dictionary containing unload status, latency_ms, and memory_release details.
        """
        pass

    @abstractmethod
    def health(self) -> dict[str, Any]:
        """Query the health and readiness status of the provider.

        Returns:
            Dictionary containing status ("healthy", "unhealthy", "not_initialized"),
            loaded state, device, and diagnostics.
        """
        pass
