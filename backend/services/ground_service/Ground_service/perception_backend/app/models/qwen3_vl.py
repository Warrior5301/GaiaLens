"""Shared Qwen3-VL backend for tools across the SAC GeoAI monorepo."""

from __future__ import annotations

import json
import re
import threading
from typing import Any

from PIL import Image

from app.core.logging import logger
from app.models.base import ModelBackend


def parse_json_object(text: str) -> dict[str, Any]:
    """Parse one JSON object from a model response, repairing syntax if needed."""
    cleaned = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "", text.strip(), flags=re.IGNORECASE
    )
    candidate = cleaned
    try:
        value = json.loads(candidate)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"Qwen response did not contain a JSON object: {text[:400]}")
        candidate = cleaned[start : end + 1]
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            try:
                from json_repair import loads as repair_json

                value = repair_json(candidate)
            except Exception as exc:
                raise ValueError(f"Qwen response was not valid JSON: {text[:400]}") from exc
            if isinstance(value, dict):
                value["_json_repaired"] = True
    if not isinstance(value, dict):
        raise ValueError("Qwen response must be a JSON object")
    return value


class QwenVLBackend(ModelBackend):
    """Lazy-loaded Transformers backend shared by VLM-capable tools.

    Tools should receive this backend through ``ModelLoader.get_or_load`` so
    separate tool instances in the same Python process reuse the model weights.
    """

    ARCH_NAME = "Qwen3-VL"

    def __init__(
        self,
        model_id: str = "qwen3_vl_8b",
        checkpoint_path: str | None = "Qwen/Qwen3-VL-8B-Instruct",
        device: str = "cuda",
        precision: str = "auto",
        hyperparameters: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            hyperparameters=hyperparameters,
        )
        self.model: Any | None = None
        self.processor: Any | None = None
        self._torch: Any | None = None
        self._inference_lock = threading.RLock()

    def load(self) -> None:
        """Load processor and weights once for this backend instance."""
        if self._is_loaded:
            return
        if not self.checkpoint_path:
            raise ValueError("Qwen3-VL requires a Hugging Face model ID in checkpoint_path")

        try:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor
        except ImportError as exc:
            raise RuntimeError(
                "Qwen3-VL dependencies are missing. Install the repo's qwen-vl extra "
                "and a PyTorch build appropriate for this machine."
            ) from exc

        logger.info("Loading Qwen3-VL model '%s' on %s", self.checkpoint_path, self.device)
        self._torch = torch
        self.processor = AutoProcessor.from_pretrained(self.checkpoint_path)
        dtype = self._resolve_dtype(torch)
        device_map: str | dict[str, str]
        if self.device == "cuda":
            device_map = "auto"
        else:
            device_map = {"": self.device}

        try:
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.checkpoint_path,
                dtype=dtype,
                device_map=device_map,
            )
            self.model.eval()
            self._is_loaded = True
        except Exception:
            self.processor = None
            self.model = None
            self._torch = None
            raise

    def _resolve_dtype(self, torch: Any) -> Any:
        if self.precision == "auto":
            return "auto"
        dtypes = {
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
            "fp32": torch.float32,
        }
        try:
            return dtypes[self.precision]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported Qwen precision '{self.precision}'. Use auto, fp16, bf16, or fp32."
            ) from exc

    def predict(
        self,
        image: Image.Image,
        prompt: str,
        task_type: str = "vqa",
        max_new_tokens: int = 512,
        temperature: float = 0.2,
        do_sample: bool = False,
        **_: Any,
    ) -> dict[str, Any]:
        """Generate text in the common ModelBackend response format."""
        if not self._is_loaded:
            self.load()
        if self.model is None or self.processor is None or self._torch is None:
            raise RuntimeError("Qwen3-VL backend did not finish loading")

        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ]}]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.model.device)
        generation_args: dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
        }
        if do_sample:
            generation_args["temperature"] = temperature

        with self._inference_lock, self._torch.inference_mode():
            generated = self.model.generate(**inputs, **generation_args)
        new_tokens = generated[:, inputs.input_ids.shape[1] :]
        raw_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return {
            "raw_text": raw_text,
            # The VLM does not provide a calibrated probability. This neutral
            # score is marked uncalibrated by consumers and is not a quality claim.
            "confidence_score": float(self.hyperparameters.get("default_confidence", 0.5)),
            "metadata": {
                **self.get_metadata(),
                "architecture": self.ARCH_NAME,
                "task_type": getattr(task_type, "value", task_type),
            },
        }

    def ask_json(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1400,
    ) -> dict[str, Any]:
        """Generate and parse a JSON object for structured tool responses."""
        response = self.predict(
            image=image,
            prompt=prompt,
            task_type="structured_output",
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )
        return parse_json_object(response["raw_text"])

    def unload(self) -> None:
        """Release model memory when the shared model cache evicts this backend."""
        if not self._is_loaded:
            return
        self.model = None
        self.processor = None
        torch = self._torch
        self._torch = None
        self._is_loaded = False
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Unloaded Qwen3-VL backend '%s'", self.model_id)


class MockQwenVLBackend(QwenVLBackend):
    """Mock backend for Qwen3-VL structured JSON inference during testing and CI."""

    def __init__(
        self,
        model_id: str = "qwen3_vl_8b",
        responses: dict[str, Any] | list[dict[str, Any]] | None = None,
        default_response: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, checkpoint_path=None, device="cpu")
        self._is_loaded = True
        self.default_response = default_response or {}
        if isinstance(responses, list):
            self._responses = list(responses)
        elif isinstance(responses, dict):
            self._responses = [responses]
        else:
            self._responses = []

    def load(self) -> None:
        self._is_loaded = True

    def ask_json(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 1400,
    ) -> dict[str, Any]:
        if self._responses:
            return self._responses.pop(0)
        return dict(self.default_response)
