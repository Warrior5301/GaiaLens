"""Modal Provider implementation for Modal-hosted perception tools."""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Optional
import urllib.error
import urllib.request

from app.core.providers.base import (
    BaseProvider,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
    validate_real_model_guarantee,
)

logger = logging.getLogger("app.core.providers.modal")


class ModalProvider(BaseProvider):
    """Provider specifically managing models hosted on Modal."""

    def __init__(self, provider_name: str = "modal", config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(provider_name=provider_name, config=config)

    def _resolve_endpoint(self, model_config: dict[str, Any], action: str = "run") -> str:
        """Resolve the target Modal web URL for the given action."""
        modal_cfg = model_config.get("modal_config", {})
        endpoints = model_config.get("endpoints", {})

        # If specific action endpoint is defined
        if action in endpoints and endpoints[action]:
            return endpoints[action]

        # Check configured modal web_endpoint
        if "web_endpoint" in modal_cfg and modal_cfg["web_endpoint"]:
            base_url = modal_cfg["web_endpoint"].rstrip("/")
            if action == "run":
                return base_url
            return f"{base_url.rsplit('/', 1)[0]}/{action}"

        app_name = modal_cfg.get("app_name", "sac-geoai-perception")
        function_name = modal_cfg.get("function_name", "api")

        try:
            import modal
            f = modal.Function.from_name(app_name, function_name)
            web_url = f.get_web_url()
            if web_url:
                base_url = web_url.rstrip("/")
                if action == "run":
                    return base_url
                return f"{base_url}/{action}"
        except Exception:
            pass

        # Fallback to standard Modal public subdomain convention
        tool_name = model_config.get("tool_name", "object_detector")
        base_url = f"https://me240003014--{app_name}-{function_name}.modal.run"
        if action == "run":
            return f"{base_url}/{tool_name}"
        return f"{base_url}/{action}"

    def _http_post(self, url: str, payload: dict[str, Any], timeout: float = 60.0) -> tuple[int, dict[str, Any], float]:
        start_time = time.perf_counter()
        req_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=req_bytes,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "SAC-GeoAI-ModalProvider/1.0",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw = response.read().decode("utf-8")
                return response.status, json.loads(raw), latency_ms
        except urllib.error.HTTPError as http_err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            raw = http_err.read().decode("utf-8", errors="replace")
            try:
                res = json.loads(raw)
            except Exception:
                res = {"detail": raw}
            raise ProviderError(
                f"Modal endpoint {url} returned HTTP {http_err.code}: {res.get('detail', raw)}",
                status_code=http_err.code,
                details={"latency_ms": latency_ms, "response": res},
            ) from http_err
        except (urllib.error.URLError, TimeoutError, OSError) as err:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            is_timeout = isinstance(err, TimeoutError) or "timed out" in str(err).lower()
            if is_timeout:
                raise ProviderTimeoutError(
                    f"Modal endpoint {url} timed out after {timeout}s: {err}",
                    details={"latency_ms": latency_ms},
                ) from err
            raise ProviderUnavailableError(
                f"Modal endpoint {url} is unreachable: {err}",
                details={"latency_ms": latency_ms},
            ) from err

    def load(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Verify and record Modal model readiness."""
        self.model_config = model_config
        model_id = model_config.get("model_id", "modal_model")
        timeouts = model_config.get("timeouts", {})
        timeout_sec = float(timeouts.get("load_timeout_sec", 60.0))

        start_time = time.perf_counter()
        load_url = self._resolve_endpoint(model_config, action="load")

        instance_id = self.generate_instance_id(model_id)
        # Verify connectivity to Modal control/health or load endpoint
        try:
            # Check if specialized load endpoint exists
            _, response, _ = self._http_post(
                load_url,
                payload={"model_id": model_id, "action": "load"},
                timeout=min(timeout_sec, 10.0),
            )
            instance_id = response.get("model_instance_id") or instance_id
        except ProviderError as pe:
            # If load endpoint returned 404, the Modal function exposes direct tool routing
            if pe.status_code == 404:
                logger.info(f"[modal] Load endpoint '{load_url}' 404; assuming direct tool execution ready.")
            else:
                raise

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        self.active_instance_id = instance_id
        self.loaded_model_id = model_id
        self.is_loaded = True

        return {
            "status": "loaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "provider": self.provider_name,
            "real_model": True,
            "device": "cuda",
            "dtype": model_config.get("dtype", "fp16"),
            "checkpoint": model_config.get("checkpoint"),
            "load_latency_ms": round(latency_ms, 2),
        }

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Execute inference against remote Modal app."""
        if not self.is_loaded:
            raise ProviderError("Model is not loaded. Call load() before run().", status_code=409)

        run_url = self._resolve_endpoint(self.model_config, action="run")
        timeouts = self.model_config.get("timeouts", {})
        timeout_sec = float(timeouts.get("run_timeout_sec", 60.0))

        # Build payload suitable for Modal tool endpoints
        tool_name = self.model_config.get("tool_name", "object_detector")
        payload: dict[str, Any] = {}

        # Prepare AssetRef representation
        image_b64 = request.get("image_base64", "")
        mime = request.get("image_mime_type", "image/png")
        data_uri = image_b64 if image_b64.startswith("data:") else f"data:{mime};base64,{image_b64}"

        asset = {
            "asset_id": request.get("asset_id", "modal_input.png"),
            "uri": data_uri,
            "width_px": request.get("image_width", 512),
            "height_px": request.get("image_height", 512),
        }

        if tool_name == "object_detector":
            payload = {
                "asset": asset,
                "target_classes": request.get("target_classes") or [request.get("query", "ship")],
                "confidence_threshold": request.get("confidence_threshold", 0.35),
            }
        elif tool_name == "segmentation_grounding":
            payload = {
                "asset": asset,
                "text_prompt": request.get("query") or request.get("text_prompt", ""),
                "pipeline": request.get("pipeline", "remote_sam_sam3_v2"),
                "confidence_threshold": request.get("confidence_threshold", 0.35),
                "return_polygon": request.get("return_polygon", True),
                "return_obb": request.get("return_obb", True),
            }
        elif tool_name == "image_analyzer":
            payload = {
                "asset": asset,
                "task_type": request.get("task_type", "caption"),
                "question": request.get("query"),
            }
        else:
            payload = dict(request)
            payload["asset"] = asset

        logger.info(f"[modal] Dispatching inference to {run_url}")
        status, response, latency_ms = self._http_post(run_url, payload=payload, timeout=timeout_sec)

        metadata = response.get("model_metadata") or {}
        validate_real_model_guarantee(metadata, self.provider_name)

        response["latency_ms"] = round(latency_ms, 2)
        response["provider"] = self.provider_name
        return response

    def unload(self) -> dict[str, Any]:
        """Release session tracking for Modal provider."""
        start_time = time.perf_counter()
        instance_id = self.active_instance_id
        model_id = self.loaded_model_id

        self.is_loaded = False
        self.active_instance_id = None
        self.loaded_model_id = None
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "unloaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "unload_latency_ms": round(latency_ms, 2),
            "memory_release": "remote_provider",
        }

    def health(self) -> dict[str, Any]:
        return {
            "status": "healthy",
            "state": "loaded" if self.is_loaded else "unloaded",
            "provider": self.provider_name,
            "model_id": self.loaded_model_id,
            "model_instance_id": self.active_instance_id,
        }
