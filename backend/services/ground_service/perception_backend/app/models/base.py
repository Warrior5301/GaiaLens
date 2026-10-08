"""Abstract base class for all neural network and vision-language model backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional


class ModelBackend(ABC):
    """Abstract interface for model execution engines."""

    def __init__(
        self,
        model_id: str,
        checkpoint_path: Optional[str] = None,
        device: str = "cpu",
        precision: str = "fp16",
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        self.model_id = model_id
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.precision = precision
        self.hyperparameters = hyperparameters or {}
        self._is_loaded = False

    @abstractmethod
    def load(self) -> None:
        """Load weights, tokenizers, and allocate GPU/CPU tensors."""
        pass

    @abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Execute forward pass inference on prepared inputs."""
        pass

    @abstractmethod
    def unload(self) -> None:
        """Release GPU memory and unload weights."""
        pass

    def is_loaded(self) -> bool:
        """Return whether the backend is ready for inference."""
        return self._is_loaded

    def get_metadata(self) -> dict[str, Any]:
        """Return runtime metadata and configuration provenance."""
        return {
            "model_id": self.model_id,
            "device": self.device,
            "precision": self.precision,
            "checkpoint_path": self.checkpoint_path,
            "is_loaded": self._is_loaded,
            "hyperparameters": self.hyperparameters,
        }
