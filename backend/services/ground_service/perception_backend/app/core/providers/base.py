"""Base provider definitions and shared helper utilities."""

from __future__ import annotations

import logging
import time
import uuid
from typing import Any, Optional
from app.core.model_provider import ModelProvider

logger = logging.getLogger("app.core.providers")

FORBIDDEN_MOCK_INDICATORS = {
    "mock",
    "mockremotesambackend",
    "mocksam3backend",
    "mockdetectorbackend",
    "dummy",
    "synthetic",
    "simulator",
}


class ProviderError(Exception):
    """Exception raised when a provider encounters a connection or runtime error."""

    def __init__(self, message: str, status_code: int = 500, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.details = details or {}


class ProviderUnavailableError(ProviderError):
    """Raised when an external or local provider is unreachable or down (HTTP 503)."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=503, details=details)


class ProviderTimeoutError(ProviderError):
    """Raised when a provider request times out (HTTP 504 / 503)."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=503, details=details)


class MockDetectedError(ProviderError):
    """Raised when a provider returns a mock or synthetic result in real mode (HTTP 503)."""

    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(message, status_code=503, details=details)


def validate_real_model_guarantee(metadata: dict[str, Any], provider_name: str) -> bool:
    """Validate that provider output does NOT use any mock or synthetic backends.

    Args:
        metadata: Model metadata dictionary.
        provider_name: Name of the active provider.

    Returns:
        True if the output satisfies the Real Model Guarantee.

    Raises:
        MockDetectedError: If mock indicators are found or real_model is False.
    """
    real_flag = metadata.get("real_model")
    if real_flag is False:
        raise MockDetectedError(
            f"Provider '{provider_name}' explicitly reported real_model=false. Mocks are prohibited."
        )

    # Inspect all metadata values for forbidden mock strings
    for k, v in metadata.items():
        if isinstance(v, str):
            v_clean = v.lower().replace("_", "").replace("-", "").replace(" ", "")
            for indicator in FORBIDDEN_MOCK_INDICATORS:
                if indicator in v_clean:
                    raise MockDetectedError(
                        f"Provider '{provider_name}' returned forbidden mock signature in '{k}': '{v}'. "
                        "Real Model Guarantee violated."
                    )

    return True


class BaseProvider(ModelProvider):
    """Base class for all provider implementations with shared logging and ID generation."""

    def __init__(self, provider_name: str, config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(provider_name=provider_name, config=config)
        self.active_instance_id: Optional[str] = None
        self.loaded_model_id: Optional[str] = None
        self.model_config: dict[str, Any] = {}
        self.is_loaded: bool = False

    def generate_instance_id(self, model_id: str) -> str:
        """Generate a unique tracking identifier for a loaded model instance."""
        return f"inst_{self.provider_name}_{model_id}_{uuid.uuid4().hex[:8]}"
