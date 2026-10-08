"""In-memory model caching with eviction and lifecycle tracking."""

from __future__ import annotations

import threading
from typing import Any
from app.core.logging import logger


class ModelCache:
    """Thread-safe in-memory cache for instantiated and loaded model backends."""

    def __init__(self, max_models: int = 5) -> None:
        self.max_models = max_models
        self._cache: dict[str, Any] = {}
        self._access_order: list[str] = []
        self._lock = threading.RLock()

    def get(self, model_key: str) -> Any | None:
        """Retrieve cached model instance, updating LRU access order."""
        with self._lock:
            if model_key in self._cache:
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                self._access_order.append(model_key)
                return self._cache[model_key]
            return None

    def set(self, model_key: str, model_instance: Any) -> None:
        """Store model instance in cache, evicting LRU model if exceeding capacity."""
        with self._lock:
            if model_key in self._cache:
                self._cache[model_key] = model_instance
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                self._access_order.append(model_key)
                return

            if len(self._cache) >= self.max_models:
                lru_key = self._access_order.pop(0)
                evicted = self._cache.pop(lru_key, None)
                if hasattr(evicted, "unload"):
                    try:
                        evicted.unload()
                    except Exception as e:
                        logger.warning(f"Error unloading evicted model '{lru_key}': {e}")
                logger.info(f"Evicted model '{lru_key}' from ModelCache (capacity: {self.max_models})")

            self._cache[model_key] = model_instance
            self._access_order.append(model_key)
            logger.info(f"Cached model '{model_key}' in ModelCache (total cached: {len(self._cache)})")

    def has(self, model_key: str) -> bool:
        """Check if model is currently cached."""
        with self._lock:
            return model_key in self._cache

    def remove(self, model_key: str) -> None:
        """Explicitly remove and unload a specific model."""
        with self._lock:
            if model_key in self._cache:
                instance = self._cache.pop(model_key)
                if model_key in self._access_order:
                    self._access_order.remove(model_key)
                if hasattr(instance, "unload"):
                    instance.unload()

    def clear(self) -> None:
        """Clear all cached models."""
        with self._lock:
            for key in list(self._cache.keys()):
                self.remove(key)
            self._cache.clear()
            self._access_order.clear()
        logger.info("Cleared all models from ModelCache")
