"""Model adapter interface connecting Image_Analyzer to EarthMindBackend."""

from __future__ import annotations

from typing import Any, Optional
from PIL import Image
from app.models.base import ModelBackend
from app.models.loaders import ModelLoader
from app.schemas.perception import GenerationConfig, ImageTaskType


class ImageAnalyzerModelInterface:
    """Encapsulates backend interaction, parameter mapping, and model lifecycle for Image_Analyzer."""

    def __init__(
        self,
        model_id: str = "earthmind_v1",
        backend: Optional[ModelBackend] = None,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        self.model_id = model_id
        self.loader = model_loader or ModelLoader()
        self._backend = backend

    @property
    def backend(self) -> ModelBackend:
        if self._backend is None:
            self._backend = self.loader.get_or_load(self.model_id)
        return self._backend

    def generate(
        self,
        image: Image.Image,
        prompt: str,
        task_type: ImageTaskType,
        generation: Optional[GenerationConfig] = None,
    ) -> dict[str, Any]:
        """Dispatch inference request to EarthMind backend."""
        gen = generation or GenerationConfig()
        return self.backend.predict(
            image=image,
            prompt=prompt,
            task_type=task_type,
            max_new_tokens=gen.max_new_tokens,
            temperature=gen.temperature,
            do_sample=gen.do_sample,
        )
