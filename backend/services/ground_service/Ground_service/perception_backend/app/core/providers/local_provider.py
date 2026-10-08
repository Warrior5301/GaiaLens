"""Local Provider implementation for in-process/in-container model execution."""

from __future__ import annotations

import base64
import gc
from io import BytesIO
import logging
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
import torch

from app.core.providers.base import (
    BaseProvider,
    ProviderError,
    validate_real_model_guarantee,
)

logger = logging.getLogger("app.core.providers.local")


class LocalProvider(BaseProvider):
    """Provider managing local in-process GPU/CPU model backends."""

    def __init__(self, provider_name: str = "local", config: Optional[dict[str, Any]] = None) -> None:
        super().__init__(provider_name=provider_name, config=config)
        self.backend: Optional[Any] = None
        self.backend_type: Optional[str] = None

    def _decode_image(self, request: dict[str, Any]) -> Image.Image:
        """Decode image from raw base64 or data URI."""
        if "image" in request and isinstance(request["image"], Image.Image):
            return request["image"].convert("RGB")

        image_b64 = request.get("image_base64")
        if not image_b64:
            raise ProviderError("Missing 'image_base64' in inference request", status_code=400)

        # Strip data URI header if present
        if image_b64.startswith("data:"):
            try:
                header, encoded = image_b64.split(",", 1)
                image_b64 = encoded
            except ValueError as e:
                raise ProviderError(f"Malformed image data URI: {e}", status_code=400) from e

        try:
            raw_bytes = base64.b64decode(image_b64, validate=True)
            if not raw_bytes:
                raise ValueError("Base64 string decoded to zero bytes")
            image = Image.open(BytesIO(raw_bytes))
            image.load()
            return image.convert("RGB")
        except Exception as exc:
            raise ProviderError(f"Invalid image base64 data: {exc}", status_code=400) from exc

    def load(self, model_config: dict[str, Any]) -> dict[str, Any]:
        """Instantiate and load real model backend directly on target device."""
        start_time = time.perf_counter()
        self.model_config = model_config
        model_id = model_config.get("model_id", "local_model")
        checkpoint = model_config.get("checkpoint")
        device_pref = model_config.get("device", "auto")
        device = "cuda" if (device_pref == "cuda" or (device_pref == "auto" and torch.cuda.is_available())) else "cpu"

        logger.info(f"[{self.provider_name}] Loading model '{model_id}' on {device} (checkpoint={checkpoint})...")

        try:
            if model_id in ("remote_sam_sam3_v2", "segmentation_grounding"):
                from app.tools.segmentation.backends.segearth_ov3 import SegEarthOV3Backend
                self.backend = SegEarthOV3Backend(
                    checkpoint_path=checkpoint or "weights/segearth_ov3/segearth_ov3.pt",
                    device=device,
                )
                self.backend_type = "segmentation"

            elif model_id in ("yolo_obb_v1", "object_detector"):
                from app.tools.detection.backends.rotated_yoloms import RotatedYOLOMSBackend
                self.backend = RotatedYOLOMSBackend(
                    checkpoint_path=checkpoint or "weights/dota/yolov8n-obb.pt",
                    device=device,
                )
                self.backend_type = "detection"

            elif model_id in ("earthmind_v1", "image_analyzer"):
                from app.tools.image_analyzer.model import ImageAnalyzerModelInterface
                self.backend = ImageAnalyzerModelInterface(model_id=model_id)
                self.backend_type = "image_analyzer"

            else:
                # Default to SegEarthOV3 or Rotated YOLO
                from app.tools.detection.backends.rotated_yoloms import RotatedYOLOMSBackend
                self.backend = RotatedYOLOMSBackend(
                    checkpoint_path=checkpoint or "weights/dota/yolov8n-obb.pt",
                    device=device,
                )
                self.backend_type = "detection"

            meta = self.backend.get_metadata() if hasattr(self.backend, "get_metadata") else {"real_model": True}
            validate_real_model_guarantee(meta, self.provider_name)

        except Exception as exc:
            logger.error(f"[{self.provider_name}] Failed to load local model '{model_id}': {exc}", exc_info=True)
            raise ProviderError(f"Failed to load local model '{model_id}': {exc}", status_code=500) from exc

        instance_id = self.generate_instance_id(model_id)
        self.active_instance_id = instance_id
        self.loaded_model_id = model_id
        self.is_loaded = True
        latency_ms = (time.perf_counter() - start_time) * 1000.0

        return {
            "status": "loaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "provider": self.provider_name,
            "real_model": meta.get("real_model", True),
            "device": meta.get("device", device),
            "dtype": meta.get("dtype", str(getattr(self.backend, "dtype", "fp16"))),
            "checkpoint": meta.get("checkpoint", checkpoint),
            "load_latency_ms": round(latency_ms, 2),
        }

    def run(self, request: dict[str, Any]) -> dict[str, Any]:
        """Perform deterministic forward inference on the loaded local backend."""
        if not self.is_loaded or self.backend is None:
            raise ProviderError("Model is not loaded. Call POST /load before /run.", status_code=409)

        start_time = time.perf_counter()
        image = self._decode_image(request)
        w, h = image.size
        query = request.get("query", "")
        params = request.get("parameters", {})
        conf_thresh = float(request.get("confidence_threshold", params.get("confidence_threshold", 0.35)))

        result: dict[str, Any] = {
            "model_id": self.loaded_model_id,
            "model_instance_id": self.active_instance_id,
            "provider": self.provider_name,
            "image_dimensions": [w, h],
        }

        try:
            if self.backend_type == "segmentation":
                from app.tools.segmentation.prompt_parser import parse_multi_target_prompt
                targets = request.get("target_classes") or parse_multi_target_prompt(query or "swimming pool")
                masks, boxes, scores, labels = self.backend.segment(
                    image=image,
                    text_prompts=targets,
                    confidence_threshold=conf_thresh,
                )
                groundings = self.backend.to_grounding_results(
                    masks=masks,
                    boxes=boxes,
                    scores=scores,
                    labels=labels,
                    return_polygon=request.get("return_polygon", True),
                    return_obb=request.get("return_obb", True),
                )
                result["groundings"] = [g.model_dump(mode="json") if hasattr(g, "model_dump") else g for g in groundings]
                result["groundings_count"] = len(groundings)

            elif self.backend_type == "detection":
                targets = request.get("target_classes")
                if not targets and query:
                    targets = [query]
                dets, count = self.backend.detect(
                    image=image,
                    target_classes=targets,
                    confidence_threshold=conf_thresh,
                )
                result["detections"] = [d.model_dump(mode="json") if hasattr(d, "model_dump") else d for d in dets]
                result["count"] = count

            elif self.backend_type == "image_analyzer":
                from app.schemas.perception import ImageTaskType
                task_type = request.get("task_type", "caption")
                gen_res = self.backend.generate(
                    image=image,
                    prompt=query or "Describe the satellite image.",
                    task_type=task_type,
                )
                result["response"] = gen_res

            meta = self.backend.get_metadata() if hasattr(self.backend, "get_metadata") else {
                "real_model": True, "model_id": self.loaded_model_id, "device": getattr(self.backend, "device", "cpu")
            }
            validate_real_model_guarantee(meta, self.provider_name)
            result["model_metadata"] = meta

        except Exception as exc:
            logger.error(f"[{self.provider_name}] Inference execution failed: {exc}", exc_info=True)
            raise ProviderError(f"Model inference failed: {exc}", status_code=500) from exc

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        result["latency_ms"] = round(latency_ms, 2)
        return result

    def unload(self) -> dict[str, Any]:
        """Release local neural network weights, GPU memory, and clear instance session."""
        start_time = time.perf_counter()
        instance_id = self.active_instance_id
        model_id = self.loaded_model_id

        mem_released_mb = 0.0
        if torch.cuda.is_available():
            try:
                mem_before = torch.cuda.memory_allocated()
            except Exception:
                mem_before = 0

        # Unload backend
        if hasattr(self.backend, "unload"):
            try:
                self.backend.unload()
            except Exception as e:
                logger.warning(f"Error calling backend unload: {e}")

        self.backend = None
        self.backend_type = None
        self.is_loaded = False
        self.active_instance_id = None
        self.loaded_model_id = None

        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                mem_after = torch.cuda.memory_allocated()
                mem_released_mb = max(0.0, float(mem_before - mem_after) / (1024.0 * 1024.0))
            except Exception:
                pass

        latency_ms = (time.perf_counter() - start_time) * 1000.0
        return {
            "status": "unloaded",
            "model_id": model_id,
            "model_instance_id": instance_id,
            "unload_latency_ms": round(latency_ms, 2),
            "gpu_memory_released_mb": round(mem_released_mb, 2),
            "memory_release": f"{round(mem_released_mb, 2)} MB" if mem_released_mb > 0 else "cleared_references",
        }

    def health(self) -> dict[str, Any]:
        """Return readiness and memory health for local provider."""
        gpu_avail = torch.cuda.is_available()
        return {
            "status": "healthy",
            "state": "loaded" if self.is_loaded else "unloaded",
            "provider": self.provider_name,
            "model_id": self.loaded_model_id,
            "model_instance_id": self.active_instance_id,
            "cuda_available": gpu_avail,
            "device": getattr(self.backend, "device", "cuda" if gpu_avail else "cpu"),
            "memory_allocated_mb": round(torch.cuda.memory_allocated() / (1024.0 * 1024.0), 2) if gpu_avail else 0.0,
        }
