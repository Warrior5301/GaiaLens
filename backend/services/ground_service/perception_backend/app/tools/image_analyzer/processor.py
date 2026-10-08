"""Input processor for Image_Analyzer preparing images and prompts."""

from __future__ import annotations

from typing import Tuple
from PIL import Image
from app.core.vision_runtime import VisionRuntime
from app.schemas.perception import ImageAnalyzerInput
from app.tools.image_analyzer.prompts import EarthMindPrompts


class ImageAnalyzerProcessor:
    """Preprocesses images and textual prompts for EarthMind inference."""

    @staticmethod
    def process(
        input_data: ImageAnalyzerInput,
        runtime: VisionRuntime,
        target_size: tuple[int, int] = (512, 512),
    ) -> Tuple[Image.Image, str]:
        """Load image, crop ROI if present, resize for model, and format prompt.

        Returns:
            Tuple of (prepared_pil_image, formatted_prompt_text)
        """
        # 1. Load image
        img = runtime.load_image(input_data.asset)

        # 2. Apply optional ROI crop
        if input_data.roi is not None:
            img, _ = runtime.crop_roi(img, input_data.roi)

        # 3. Standardize dimensions for model input
        prepared_img = runtime.resize_for_model(img, target_size=target_size)

        # 4. Format prompt based on task type
        if input_data.task_type == "caption":
            prompt = EarthMindPrompts.format_caption_prompt(input_data.question)
        elif input_data.task_type == "vqa":
            # Semantic validation ensured question is present
            prompt = EarthMindPrompts.format_vqa_prompt(input_data.question or "")
        else:
            prompt = EarthMindPrompts.format_scene_prompt()

        return prepared_img, prompt
