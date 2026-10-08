"""Lifecycle Controller orchestrating hosting-agnostic model operations."""

from __future__ import annotations

import base64
from io import BytesIO
import logging
from pathlib import Path
import time
from typing import Any, Optional
from PIL import Image
import yaml

from app.api.schemas import (
    HealthResponse,
    LoadRequest,
    LoadResponse,
    RunRequest,
    RunResponse,
    SmokeTestRequest,
    SmokeTestResponse,
    UnloadRequest,
    UnloadResponse,
)
from app.core.model_provider import ModelProvider
from app.core.model_registry import ModelState, model_registry
from app.core.providers import (
    BaseProvider,
    HTTPProvider,
    LocalProvider,
    MockDetectedError,
    ModalProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger("app.core.lifecycle_controller")

TOOL_TO_MODEL_MAP = {
    "object_detector": "yolo_obb_v1",
    "segmentation_grounding": "remote_sam_sam3_v2",
    "image_analyzer": "earthmind_v1",
}

MODEL_TO_TOOL_MAP = {v: k for k, v in TOOL_TO_MODEL_MAP.items()}


class LifecycleController:
    """Orchestrates model lifecycle transitions and provider dispatching."""

    def __init__(self, config_path: str = "config/model_runtime.yaml") -> None:
        self.config_path = Path(config_path)
        self.runtime_config: dict[str, Any] = self._load_config()
        self._providers: dict[str, ModelProvider] = {}
        self._init_providers()

    def _load_config(self) -> dict[str, Any]:
        """Load model runtime YAML configuration."""
        if not self.config_path.exists():
            logger.warning(f"Config path '{self.config_path}' not found, using empty runtime defaults.")
            return {"default_provider": "local", "models": {}}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
        except Exception as e:
            logger.error(f"Failed to parse runtime config '{self.config_path}': {e}")
            return {"default_provider": "local", "models": {}}

    def _init_providers(self) -> None:
        """Instantiate available hosting providers."""
        self._providers["local"] = LocalProvider(provider_name="local")
        self._providers["http"] = HTTPProvider(provider_name="http")
        self._providers["modal"] = ModalProvider(provider_name="modal")

    def resolve_model_id(self, identifier: str) -> str:
        """Resolve either a tool name or a model ID to the canonical model_id."""
        return TOOL_TO_MODEL_MAP.get(identifier, identifier)

    def get_model_config(self, model_id: str) -> dict[str, Any]:
        """Retrieve model configuration for specified model_id."""
        canonical_id = self.resolve_model_id(model_id)
        models = self.runtime_config.get("models", {})

        if canonical_id in models:
            cfg = dict(models[canonical_id])
            cfg["model_id"] = canonical_id
            return cfg

        # Check tool name alias in config keys
        tool_name = MODEL_TO_TOOL_MAP.get(canonical_id)
        if tool_name and tool_name in models:
            cfg = dict(models[tool_name])
            cfg["model_id"] = canonical_id
            return cfg

        # Fallback default configuration
        default_prov = self.runtime_config.get("default_provider", "local")
        return {
            "model_id": canonical_id,
            "provider": default_prov,
            "device": "auto",
            "real_model": True,
        }

    def get_provider(self, provider_name: str) -> ModelProvider:
        """Resolve provider instance by name."""
        prov = self._providers.get(provider_name.lower())
        if not prov:
            raise ProviderError(f"Unsupported provider '{provider_name}'. Supported: {list(self._providers.keys())}", status_code=400)
        return prov

    def validate_image_base64(self, b64_str: str) -> tuple[int, int, str]:
        """Validate base64 string and return image (width, height, format)."""
        if not b64_str or not isinstance(b64_str, str):
            raise ProviderError("image_base64 must be a non-empty string", status_code=400)

        cleaned = b64_str
        mime = "image/png"
        if b64_str.startswith("data:"):
            try:
                header, encoded = b64_str.split(",", 1)
                cleaned = encoded
                if "image/jpeg" in header or "image/jpg" in header:
                    mime = "image/jpeg"
            except ValueError as e:
                raise ProviderError(f"Malformed image data URI: {e}", status_code=400) from e

        try:
            raw_bytes = base64.b64decode(cleaned, validate=True)
            if not raw_bytes:
                raise ValueError("Base64 string decoded to zero bytes")
            with Image.open(BytesIO(raw_bytes)) as img:
                img_format = (img.format or "PNG").upper()
                if img_format not in ("PNG", "JPEG", "JPG"):
                    raise ProviderError(f"Unsupported image format '{img_format}'. Only PNG and JPEG are supported.", status_code=400)
                width, height = img.size
                if width <= 0 or height <= 0:
                    raise ProviderError(f"Invalid image dimensions: {width}x{height}", status_code=400)
                return width, height, mime
        except Exception as exc:
            if isinstance(exc, ProviderError):
                raise
            raise ProviderError(f"Invalid image data: {exc}", status_code=400) from exc

    def load_model(self, req: LoadRequest) -> LoadResponse:
        """Handle POST /load request."""
        model_id = self.resolve_model_id(req.model_id)
        config = self.get_model_config(model_id)
        provider_name = (req.provider or config.get("provider") or self.runtime_config.get("default_provider", "local")).lower()

        logger.info(f"[LifecycleController] Loading model '{model_id}' via provider '{provider_name}'...")
        provider = self.get_provider(provider_name)

        model_registry.set_loading(model_id=model_id, provider=provider_name)
        try:
            load_result = provider.load(config)
            instance_id = load_result.get("model_instance_id") or f"inst_{provider_name}_{model_id}"

            session = model_registry.set_loaded(
                model_id=model_id,
                provider=provider_name,
                instance_id=instance_id,
                metadata=load_result,
            )

            smoke_test_res = None
            if req.run_startup_test:
                logger.info(f"[LifecycleController] Executing startup test for '{model_id}'...")
                smoke_test_res = self._execute_test_run(
                    model_id=model_id,
                    provider=provider,
                    custom_image=req.test_image_base64,
                    custom_query=req.test_query,
                )

            return LoadResponse(
                status="loaded",
                model_id=model_id,
                model_instance_id=instance_id,
                provider=provider_name,
                real_model=load_result.get("real_model", True),
                load_latency_ms=load_result.get("load_latency_ms", 0.0),
                device=load_result.get("device"),
                dtype=load_result.get("dtype"),
                checkpoint=load_result.get("checkpoint"),
                smoke_test=smoke_test_res,
            )

        except Exception as exc:
            model_registry.set_error(model_id, str(exc))
            logger.error(f"[LifecycleController] Load failed for '{model_id}': {exc}", exc_info=True)
            raise

    def run_model(self, req: RunRequest) -> RunResponse:
        """Handle POST /run request."""
        model_id = self.resolve_model_id(req.model_id)

        # 1. Lifecycle verification: Check if model is loaded
        if not model_registry.is_loaded(model_id):
            raise ProviderError(
                f"Model '{model_id}' is not loaded. Call POST /load before /run.",
                status_code=409,
            )

        session = model_registry.get(model_id)
        if not session:
            raise ProviderError(f"No active session found for '{model_id}'", status_code=409)

        provider = self.get_provider(session.provider)

        # 2. Strict input image validation
        width, height, mime = self.validate_image_base64(req.image_base64)

        # 3. Build execution payload
        payload = req.model_dump(exclude_none=True)
        payload["image_width"] = width
        payload["image_height"] = height
        payload["image_mime_type"] = mime
        payload["model_instance_id"] = session.model_instance_id

        # 4. Dispatch inference
        model_registry.set_running(model_id)
        try:
            raw_res = provider.run(payload)
            model_registry.set_run_finished(model_id)

            metadata = raw_res.get("model_metadata") or {}
            from app.core.providers.base import validate_real_model_guarantee
            validate_real_model_guarantee(metadata, session.provider)
            real_model = metadata.get("real_model", session.real_model)

            return RunResponse(
                status="success",
                model_id=model_id,
                model_instance_id=session.model_instance_id,
                provider=session.provider,
                real_model=real_model,
                latency_ms=raw_res.get("latency_ms", 0.0),
                image_dimensions=[width, height],
                detections=raw_res.get("detections"),
                count=raw_res.get("count"),
                groundings=raw_res.get("groundings"),
                groundings_count=raw_res.get("groundings_count"),
                response=raw_res.get("response"),
                model_metadata=metadata,
                processing_metrics=raw_res.get("processing_metrics"),
            )
        except Exception as exc:
            model_registry.set_run_finished(model_id)
            raise

    def unload_model(self, req: UnloadRequest) -> UnloadResponse:
        """Handle POST /unload request."""
        model_id = None
        if req.model_id:
            model_id = self.resolve_model_id(req.model_id)
        elif req.model_instance_id:
            sess = model_registry.get_by_instance_id(req.model_instance_id)
            if sess:
                model_id = sess.model_id

        if not model_id:
            raise ProviderError("Must supply 'model_id' or 'model_instance_id' to /unload", status_code=400)

        session = model_registry.get(model_id)
        if not session or session.state == ModelState.UNLOADED:
            return UnloadResponse(
                status="unloaded",
                model_id=model_id,
                model_instance_id=req.model_instance_id,
                unload_latency_ms=0.0,
                memory_release="already_unloaded",
            )

        provider = self.get_provider(session.provider)
        model_registry.set_unloading(model_id)

        try:
            unload_result = provider.unload()
            model_registry.set_unloaded(model_id)

            return UnloadResponse(
                status="unloaded",
                model_id=model_id,
                model_instance_id=unload_result.get("model_instance_id", session.model_instance_id),
                unload_latency_ms=unload_result.get("unload_latency_ms", 0.0),
                gpu_memory_released_mb=unload_result.get("gpu_memory_released_mb"),
                memory_release=unload_result.get("memory_release", "remote_provider"),
            )
        except Exception as exc:
            model_registry.set_error(model_id, f"Unload failed: {exc}")
            raise

    def health(self) -> HealthResponse:
        """Handle GET /health request."""
        models_health: dict[str, dict[str, Any]] = {}
        configured_models = self.runtime_config.get("models", {})

        for m_id, cfg in configured_models.items():
            canonical_id = self.resolve_model_id(m_id)
            sess = model_registry.get(canonical_id)
            if sess:
                models_health[canonical_id] = {
                    "state": sess.state.value if isinstance(sess.state, ModelState) else str(sess.state),
                    "provider": sess.provider,
                    "real_model": sess.real_model,
                    "device": sess.device,
                    "model_instance_id": sess.model_instance_id,
                }
            else:
                models_health[canonical_id] = {
                    "state": "unloaded",
                    "provider": cfg.get("provider", "local"),
                }

        return HealthResponse(
            status="healthy",
            service="sac-geoai-perception",
            models=models_health,
        )

    def smoke_test(self, req: SmokeTestRequest) -> SmokeTestResponse:
        """Handle POST /smoke_test: full reproducible cycle (load -> run -> unload)."""
        model_id = self.resolve_model_id(req.model_id)

        # 1. Internal Load
        load_res = self.load_model(LoadRequest(model_id=model_id, run_startup_test=False))

        # 2. Internal Run
        test_img_b64 = req.image_base64
        if not test_img_b64:
            default_img_path = Path("test_assets/default_satellite.png")
            if not default_img_path.exists():
                default_img_path = Path("examples/real_satellite/05873_0000.png")
            with open(default_img_path, "rb") as f:
                test_img_b64 = base64.b64encode(f.read()).decode("ascii")

        query = req.query or "Locate the swimming pool."
        run_res = self.run_model(RunRequest(model_id=model_id, image_base64=test_img_b64, query=query))

        # 3. Internal Unload (CRITICAL: Smoke test must NOT leave model loaded)
        unload_res = self.unload_model(UnloadRequest(model_id=model_id))

        return SmokeTestResponse(
            status="passed",
            model_id=model_id,
            provider=load_res.provider,
            real_model=load_res.real_model and run_res.real_model,
            load=load_res.model_dump(mode="json"),
            run=run_res.model_dump(mode="json"),
            unload=unload_res.model_dump(mode="json"),
            result={
                "detections_count": run_res.count or len(run_res.detections or []),
                "groundings_count": run_res.groundings_count or len(run_res.groundings or []),
                "has_response": run_res.response is not None,
            },
        )

    def _execute_test_run(
        self,
        model_id: str,
        provider: ModelProvider,
        custom_image: Optional[str] = None,
        custom_query: Optional[str] = None,
    ) -> dict[str, Any]:
        """Helper to run a test inference query without altering lifecycle state."""
        start_time = time.perf_counter()
        img_b64 = custom_image
        if not img_b64:
            default_img_path = Path("test_assets/default_satellite.png")
            if not default_img_path.exists():
                default_img_path = Path("examples/real_satellite/05873_0000.png")
            with open(default_img_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode("ascii")

        w, h, mime = self.validate_image_base64(img_b64)
        query = custom_query or "Locate the swimming pool."

        test_payload = {
            "model_id": model_id,
            "image_base64": img_b64,
            "query": query,
            "image_width": w,
            "image_height": h,
            "image_mime_type": mime,
        }
        res = provider.run(test_payload)
        latency = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "passed",
            "latency_ms": round(latency, 2),
            "output_summary": {
                "detections": len(res.get("detections", [])),
                "groundings": len(res.get("groundings", [])),
            },
        }


# Global controller instance
lifecycle_controller = LifecycleController()
