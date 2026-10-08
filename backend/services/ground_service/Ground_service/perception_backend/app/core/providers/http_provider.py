"""HTTP Provider implementation for arbitrary externally hosted model services."""

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

logger = logging.getLogger("app.core.providers.http")


class HTTPProvider(BaseProvider):
    """Provider for remote HTTP-based model hosting (RunPod, Kubernetes, external endpoints)."""

    def __init__(self, provider_name: str = "http", config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(provider_name=provider_name, config=config)

    def _http_request(
        self,
        url: str,
        method: str = "POST",
        payload: Optional[dict[str, Any]] = None,
        timeout: float = 60.0,
        max_retries: int = 0,
    ) -> tuple[int, dict[str, Any], float]:
        """Perform an HTTP request with timing, bounded retries, and structured error handling."""
        data_bytes = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "SAC-GeoAI-LifecycleController/1.0",
        }

        attempts = 0
        last_error: Optional[Exception] = None

        while attempts <= max_retries:
            attempts += 1
            start_time = time.perf_counter()
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method=method)

            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    latency_ms = (time.perf_counter() - start_time) * 1000.0
                    raw_body = response.read().decode("utf-8")
                    try:
                        res_json = json.loads(raw_body)
                    except json.JSONDecodeError:
                        res_json = {"raw_text": raw_body}
                    return response.status, res_json, latency_ms

            except urllib.error.HTTPError as http_err:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                raw_body = http_err.read().decode("utf-8", errors="replace")
                try:
                    error_json = json.loads(raw_body)
                except Exception:
                    error_json = {"detail": raw_body}

                # Non-retriable client errors
                if http_err.code in (400, 404, 409, 422):
                    raise ProviderError(
                        message=f"HTTP {http_err.code} from {url}: {error_json.get('detail', raw_body)}",
                        status_code=http_err.code,
                        details={"endpoint": url, "response": error_json, "latency_ms": latency_ms},
                    ) from http_err

                last_error = ProviderError(
                    message=f"HTTP {http_err.code} from {url}: {raw_body}",
                    status_code=http_err.code,
                    details={"endpoint": url, "response": error_json, "latency_ms": latency_ms},
                )

            except (urllib.error.URLError, TimeoutError, OSError) as conn_err:
                latency_ms = (time.perf_counter() - start_time) * 1000.0
                is_timeout = isinstance(conn_err, TimeoutError) or "timed out" in str(conn_err).lower()
                if is_timeout:
                    last_error = ProviderTimeoutError(
                        message=f"Request to {url} timed out after {timeout:.1f}s: {conn_err}",
                        details={"endpoint": url, "latency_ms": latency_ms, "timeout": timeout},
                    )
                else:
                    last_error = ProviderUnavailableError(
                        message=f"Could not connect to provider endpoint {url}: {conn_err}",
                        details={"endpoint": url, "latency_ms": latency_ms},
                    )

            if attempts <= max_retries:
                time.sleep(0.5 * attempts)

        if isinstance(last_error, ProviderError):
            raise last_error
        raise ProviderUnavailableError(f"Request to {url} failed after {attempts} attempts: {last_error}")

    def load(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Request external HTTP model host to load model."""
        self.model_config = model_config
        model_id = model_config.get("model_id", "unknown_model")
        endpoints = model_config.get("endpoints", {})
        load_url = endpoints.get("load")
        timeouts = model_config.get("timeouts", {})
        timeout_sec = float(timeouts.get("load_timeout_sec", 120.0))

        if not load_url:
            raise ProviderError(f"Missing 'load' endpoint URL in config for model '{model_id}'")

        payload = {
            "model_id": model_id,
            "checkpoint": model_config.get("checkpoint"),
            "device": model_config.get("device", "auto"),
            "dtype": model_config.get("dtype", "fp16"),
        }

        logger.info(f"[{self.provider_name}] Calling load endpoint: {load_url} (timeout={timeout_sec}s)")
        status, response, latency_ms = self._http_request(
            url=load_url,
            method="POST",
            payload=payload,
            timeout=timeout_sec,
            max_retries=0,  # Do not blindly retry load
        )

        metadata = response.get("model_metadata") or response
        validate_real_model_guarantee(metadata, self.provider_name)

        instance_id = response.get("model_instance_id") or self.generate_instance_id(model_id)
        self.active_instance_id = instance_id
        self.loaded_model_id = model_id
        self.is_loaded = True

        return {
            "status": "loaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "provider": self.provider_name,
            "real_model": metadata.get("real_model", True),
            "device": metadata.get("device", "remote"),
            "dtype": metadata.get("dtype", "fp16"),
            "checkpoint": metadata.get("checkpoint", model_config.get("checkpoint")),
            "load_latency_ms": round(latency_ms, 2),
            "raw_response": response,
        }

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Send inference request to external HTTP model runner."""
        if not self.is_loaded:
            raise ProviderError("Model is not loaded. Call load() before run().", status_code=409)

        endpoints = self.model_config.get("endpoints", {})
        run_url = endpoints.get("run")
        timeouts = self.model_config.get("timeouts", {})
        timeout_sec = float(timeouts.get("run_timeout_sec", 60.0))
        max_retries = int(timeouts.get("max_retries", 1))

        if not run_url:
            raise ProviderError(f"Missing 'run' endpoint URL for model '{self.loaded_model_id}'")

        # Map generic request format to remote provider payload
        payload = dict(request)
        if self.active_instance_id and "model_instance_id" not in payload:
            payload["model_instance_id"] = self.active_instance_id

        # Normalize query / text_prompt for compatibility
        if "query" in payload and "text_prompt" not in payload:
            payload["text_prompt"] = payload["query"]

        logger.info(f"[{self.provider_name}] Calling run endpoint: {run_url} (timeout={timeout_sec}s)")
        status, response, latency_ms = self._http_request(
            url=run_url,
            method="POST",
            payload=payload,
            timeout=timeout_sec,
            max_retries=max_retries,
        )

        # Enforce Real Model Guarantee on execution results
        metadata = response.get("model_metadata") or {}
        validate_real_model_guarantee(metadata, self.provider_name)

        response["latency_ms"] = round(latency_ms, 2)
        response["provider"] = self.provider_name
        return response

    def unload(self) -> dict[str, Any]:
        """Request external HTTP model host to release model."""
        if not self.is_loaded:
            return {
                "status": "unloaded",
                "model_id": self.loaded_model_id,
                "model_instance_id": self.active_instance_id,
                "unload_latency_ms": 0.0,
                "memory_release": "remote_provider",
            }

        endpoints = self.model_config.get("endpoints", {})
        unload_url = endpoints.get("unload")
        timeouts = self.model_config.get("timeouts", {})
        timeout_sec = float(timeouts.get("unload_timeout_sec", 30.0))

        start_time = time.perf_counter()
        if unload_url:
            payload = {
                "model_id": self.loaded_model_id,
                "model_instance_id": self.active_instance_id,
            }
            try:
                _, _, _ = self._http_request(
                    url=unload_url,
                    method="POST",
                    payload=payload,
                    timeout=timeout_sec,
                    max_retries=0,  # Never repeat a state-changing unload blindly
                )
            except Exception as e:
                logger.warning(f"[{self.provider_name}] Remote unload call warning: {e}")

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        instance_id = self.active_instance_id
        model_id = self.loaded_model_id

        self.is_loaded = False
        self.active_instance_id = None
        self.loaded_model_id = None

        return {
            "status": "unloaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "unload_latency_ms": round(latency_ms, 2),
            "memory_release": "remote_provider",
        }

    def health(self) -> dict[str, Any]:
        """Query remote health endpoint."""
        endpoints = self.model_config.get("endpoints", {})
        health_url = endpoints.get("health")

        if not health_url:
            return {
                "status": "healthy" if self.is_loaded else "unloaded",
                "state": "loaded" if self.is_loaded else "unloaded",
                "provider": self.provider_name,
                "model_id": self.loaded_model_id,
                "model_instance_id": self.active_instance_id,
            }

        try:
            _, response, latency_ms = self._http_request(
                url=health_url, method="GET", timeout=5.0, max_retries=0
            )
            return {
                "status": response.get("status", "healthy"),
                "state": "loaded" if self.is_loaded else "unloaded",
                "provider": self.provider_name,
                "model_id": self.loaded_model_id,
                "model_instance_id": self.active_instance_id,
                "latency_ms": round(latency_ms, 2),
                "remote_details": response,
            }
        except Exception as exc:
            return {
                "status": "unhealthy",
                "state": "error",
                "provider": self.provider_name,
                "model_id": self.loaded_model_id,
                "error": str(exc),
            }
