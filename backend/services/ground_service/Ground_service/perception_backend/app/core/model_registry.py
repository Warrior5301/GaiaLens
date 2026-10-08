"""Thread-safe model registry and lifecycle session state management."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
import logging
import threading
from typing import Any, Optional

logger = logging.getLogger("app.core.model_registry")


class ModelState(str, Enum):
    """Lifecycle states for hosted model instances."""
    UNLOADED = "UNLOADED"
    LOADING = "LOADING"
    LOADED = "LOADED"
    RUNNING = "RUNNING"
    UNLOADING = "UNLOADING"
    ERROR = "ERROR"


@dataclass
class ModelSession:
    """Represents the tracking session state for a model."""
    model_id: str
    provider: str
    state: ModelState = ModelState.UNLOADED
    model_instance_id: Optional[str] = None
    loaded_at: Optional[str] = None
    last_used_at: Optional[str] = None
    real_model: bool = True
    device: Optional[str] = None
    dtype: Optional[str] = None
    checkpoint: Optional[str] = None
    metadata: dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert session to standardized dictionary."""
        return {
            "model_id": self.model_id,
            "provider": self.provider,
            "state": self.state.value if isinstance(self.state, ModelState) else str(self.state),
            "model_instance_id": self.model_instance_id,
            "loaded_at": self.loaded_at,
            "last_used_at": self.last_used_at,
            "real_model": self.real_model,
            "device": self.device,
            "dtype": self.dtype,
            "checkpoint": self.checkpoint,
            "metadata": self.metadata,
            "error_message": self.error_message,
        }


class ModelRegistry:
    """Thread-safe registry managing model lifecycle states and active provider sessions."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, ModelSession] = {}
        self._instance_to_model: dict[str, str] = {}

    def get_or_create(self, model_id: str, provider: str = "unknown") -> ModelSession:
        """Retrieve existing model session or create in UNLOADED state."""
        with self._lock:
            if model_id not in self._sessions:
                self._sessions[model_id] = ModelSession(
                    model_id=model_id,
                    provider=provider,
                    state=ModelState.UNLOADED,
                )
            return self._sessions[model_id]

    def get(self, model_id: str) -> Optional[ModelSession]:
        """Get model session by model_id."""
        with self._lock:
            return self._sessions.get(model_id)

    def get_by_instance_id(self, instance_id: str) -> Optional[ModelSession]:
        """Find active model session by instance_id."""
        with self._lock:
            model_id = self._instance_to_model.get(instance_id)
            if model_id:
                return self._sessions.get(model_id)
            for sess in self._sessions.values():
                if sess.model_instance_id == instance_id:
                    return sess
            return None

    def is_loaded(self, model_id: str) -> bool:
        """Check if model is currently in LOADED or RUNNING state."""
        with self._lock:
            sess = self._sessions.get(model_id)
            return sess is not None and sess.state in (ModelState.LOADED, ModelState.RUNNING)

    def set_loading(self, model_id: str, provider: str) -> ModelSession:
        """Mark model as LOADING."""
        with self._lock:
            sess = self.get_or_create(model_id, provider=provider)
            sess.provider = provider
            sess.state = ModelState.LOADING
            sess.error_message = None
            return sess

    def set_loaded(
        self,
        model_id: str,
        provider: str,
        instance_id: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> ModelSession:
        """Mark model as LOADED with instance identifier and timestamp."""
        with self._lock:
            sess = self.get_or_create(model_id, provider=provider)
            sess.provider = provider
            sess.state = ModelState.LOADED
            sess.model_instance_id = instance_id
            sess.loaded_at = datetime.now(timezone.utc).isoformat()
            sess.last_used_at = sess.loaded_at
            sess.metadata = metadata or {}
            sess.real_model = sess.metadata.get("real_model", True)
            sess.device = sess.metadata.get("device")
            sess.dtype = sess.metadata.get("dtype")
            sess.checkpoint = sess.metadata.get("checkpoint")
            sess.error_message = None

            self._instance_to_model[instance_id] = model_id
            logger.info(f"[ModelRegistry] Model '{model_id}' marked LOADED (instance='{instance_id}', provider='{provider}')")
            return sess

    def set_running(self, model_id: str) -> ModelSession:
        """Record inference execution activity and mark RUNNING."""
        with self._lock:
            sess = self.get(model_id)
            if not sess:
                raise ValueError(f"Model '{model_id}' not found in registry")
            sess.state = ModelState.RUNNING
            sess.last_used_at = datetime.now(timezone.utc).isoformat()
            return sess

    def set_run_finished(self, model_id: str) -> ModelSession:
        """Restore state to LOADED after inference completion."""
        with self._lock:
            sess = self.get(model_id)
            if sess and sess.state == ModelState.RUNNING:
                sess.state = ModelState.LOADED
            return sess

    def set_unloading(self, model_id: str) -> ModelSession:
        """Mark model as UNLOADING."""
        with self._lock:
            sess = self.get(model_id)
            if sess:
                sess.state = ModelState.UNLOADING
            return sess

    def set_unloaded(self, model_id: str) -> ModelSession:
        """Mark model as UNLOADED and clear instance mapping."""
        with self._lock:
            sess = self.get(model_id)
            if sess:
                old_inst = sess.model_instance_id
                if old_inst and old_inst in self._instance_to_model:
                    del self._instance_to_model[old_inst]

                sess.state = ModelState.UNLOADED
                sess.model_instance_id = None
                sess.metadata = {}
                logger.info(f"[ModelRegistry] Model '{model_id}' marked UNLOADED")
            return sess

    def set_error(self, model_id: str, error_message: str) -> ModelSession:
        """Mark model state as ERROR with diagnostic details."""
        with self._lock:
            sess = self.get_or_create(model_id)
            sess.state = ModelState.ERROR
            sess.error_message = error_message
            logger.error(f"[ModelRegistry] Model '{model_id}' entered ERROR state: {error_message}")
            return sess

    def list_sessions(self) -> dict[str, dict[str, Any]]:
        """Return snapshot of all registered model sessions."""
        with self._lock:
            return {m_id: sess.to_dict() for m_id, sess in self._sessions.items()}


# Global singleton instance
model_registry = ModelRegistry()
