"""Model loader service instantiating backends from ModelDefinitions."""

from __future__ import annotations

import threading
from typing import Optional
from app.core.device import DeviceManager
from app.core.logging import logger
from app.core.model_cache import ModelCache
from app.models.base import ModelBackend
from app.models.earthmind import EarthMindBackend
from app.models.registry import ModelDefinition, ModelRegistry

_SHARED_MODEL_CACHE = ModelCache()
_MODEL_LOAD_LOCK = threading.RLock()


class ModelLoader:
    """Instantiates and configures ModelBackend instances from declarative registry definitions."""

    def __init__(
        self,
        registry: Optional[ModelRegistry] = None,
        cache: Optional[ModelCache] = None,
    ) -> None:
        self.registry = registry or ModelRegistry("config/model_registry.yaml")
        # Every default loader in this process shares one cache, so separately
        # constructed team tools reuse the same heavyweight backend instance.
        self.cache = cache if cache is not None else _SHARED_MODEL_CACHE

    def get_or_load(
        self,
        model_id: str,
        device_override: Optional[str] = None,
    ) -> ModelBackend:
        """Retrieve model from cache or instantiate and load it."""
        model_def = self.registry.get_model_definition(model_id)
        target_device = device_override or model_def.device or DeviceManager.get_preferred_device()
        cache_key = f"{model_id}|{target_device}|{model_def.precision}"
        with _MODEL_LOAD_LOCK:
            cached = self.cache.get(cache_key)
            if cached is not None and isinstance(cached, ModelBackend) and cached.is_loaded():
                return cached

            backend = self._instantiate_backend(model_def, target_device)
            backend.load()
            self.cache.set(cache_key, backend)
            logger.info(f"Loaded and cached backend for model_id '{model_id}' on {target_device}")
            return backend

    def _instantiate_backend(self, model_def: ModelDefinition, device: str) -> ModelBackend:
        arch = (model_def.architecture or "").lower()
        task = (model_def.task or "").lower()

        if "qwen3-vl" in arch or "qwen3_vl" in model_def.model_id.lower():
            from app.models.qwen3_vl import QwenVLBackend

            return QwenVLBackend(
                model_id=model_def.model_id,
                checkpoint_path=model_def.weights.uri if model_def.weights else None,
                device=device,
                precision=model_def.precision,
                hyperparameters=model_def.hyperparameters,
            )

        if "earthmind" in arch or "caption" in task:
            return EarthMindBackend(
                model_id=model_def.model_id,
                checkpoint_path=model_def.weights.uri if model_def.weights else None,
                device=device,
                precision=model_def.precision,
                hyperparameters=model_def.hyperparameters,
            )

        if "yolo" in arch or "object_detection" in task or "detector" in task:
            from app.tools.object_detector.backend import RemoteSensingYOLOBackend
            return RemoteSensingYOLOBackend(
                model_id=model_def.model_id,
                checkpoint_path=model_def.weights.uri if model_def.weights else None,
                device=device,
                precision=model_def.precision,
                hyperparameters=model_def.hyperparameters,
            )

        if "mock_detector" in arch or "mock_detector" in model_def.model_id:
            from app.tools.object_detector.backend import MockDetectorBackend
            return MockDetectorBackend(
                model_id=model_def.model_id,
                device=device,
                precision=model_def.precision,
                hyperparameters=model_def.hyperparameters,
            )

        raise ValueError(
            f"Unsupported architecture '{model_def.architecture}' for task '{model_def.task}'"
        )
