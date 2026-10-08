"""
Temporal Comparison Tool
========================

A modular tool for multi-temporal remote-sensing image comparison and change detection.
Uses the EarthDial model to analyze changes between 2 or more satellite/aerial observations.
"""

import base64
import io
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import requests
from PIL import Image

from model import EarthDialModel
from schemas import (
    ImageResolutionInfo,
    PromptStyle,
    PromptTemplateItem,
    TemporalComparisonRequest,
    TemporalComparisonResponse,
)

logger = logging.getLogger("TemporalComparisonTool")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")


class TemporalComparisonTool:
    """
    Modular tool for temporal image change detection and visual reasoning.
    """

    def __init__(
        self,
        model: Optional[EarthDialModel] = None,
        checkpoint: str = "akshaydudhane/EarthDial_4B_RGB",
        device: Optional[str] = None,
        mock_mode: bool = False,
        **model_kwargs
    ):
        """
        Initialize the temporal comparison tool.

        Args:
            model: Optional pre-initialized EarthDialModel instance.
            checkpoint: Model checkpoint to load if no model instance is passed.
            device: 'cuda', 'cpu', or auto-detect.
            mock_mode: Whether to run in mock simulation mode.
        """
        if model is not None:
            self.model = model
        else:
            self.model = EarthDialModel(
                checkpoint=checkpoint,
                device=device,
                mock_mode=mock_mode,
                **model_kwargs
            )

    @staticmethod
    def load_image(image_input: Union[str, Path, Image.Image, bytes, io.BytesIO]) -> Image.Image:
        """
        Loads an image from various input types:
        - PIL Image
        - Local file path
        - Web URL (http/https)
        - Base64 data string (e.g. data:image/png;base64,... or raw base64)
        - Raw bytes or BytesIO
        """
        if isinstance(image_input, Image.Image):
            return image_input.convert("RGB")

        if isinstance(image_input, (bytes, io.BytesIO)):
            stream = io.BytesIO(image_input) if isinstance(image_input, bytes) else image_input
            return Image.open(stream).convert("RGB")

        if isinstance(image_input, (str, Path)):
            image_str = str(image_input).strip()

            # 1. Web URL
            if image_str.startswith("http://") or image_str.startswith("https://"):
                resp = requests.get(image_str, timeout=30)
                resp.raise_for_status()
                return Image.open(io.BytesIO(resp.content)).convert("RGB")

            # 2. Base64
            if image_str.startswith("data:image/") or len(image_str) > 500:
                if "," in image_str:
                    image_str = image_str.split(",", 1)[1]
                try:
                    raw = base64.b64decode(image_str)
                    return Image.open(io.BytesIO(raw)).convert("RGB")
                except Exception:
                    pass

            # 3. Local file path
            path = Path(image_str)
            if path.exists():
                return Image.open(path).convert("RGB")

            # Fallback: maybe raw base64 without data header
            try:
                raw = base64.b64decode(image_str)
                return Image.open(io.BytesIO(raw)).convert("RGB")
            except Exception as e:
                raise ValueError(f"Could not load image from input '{image_str[:60]}...': {e}")

        raise TypeError(f"Unsupported image input type: {type(image_input)}")

    @staticmethod
    def format_prompt(
        num_images: int,
        query: str,
        style: PromptStyle = PromptStyle.EVALUATION,
        task_token: Optional[str] = "[changedet] [hr_rgb_temp_0.5]"
    ) -> str:
        """
        Formats prompt with required multi-image tokens.
        - Evaluation: [changedet] [hr_rgb_temp_0.5] <image> <image> ... \\n {query}
        - Conversational: Image-1: <image>\\nImage-2: <image>\\n... {query}
        """
        if "<image>" in query or style == PromptStyle.RAW:
            return query

        if style == PromptStyle.CONVERSATIONAL:
            prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(num_images))
            return f"{prefix}{query.strip()}"

        # Default Evaluation style
        image_tags = " ".join(["<image>"] * num_images)
        token_prefix = f"{task_token.strip()} " if task_token and task_token.strip() else ""
        return f"{token_prefix}{image_tags}\n {query.strip()}"

    def compare(
        self,
        images: List[Union[str, Path, Image.Image, bytes]],
        query: str = "Analyze the images and identify all semantic and structural changes between the observations.",
        prompt_style: PromptStyle = PromptStyle.EVALUATION,
        task_token: Optional[str] = "[changedet] [hr_rgb_temp_0.5]",
        temperature: float = 0.0,
        max_new_tokens: int = 256,
        top_p: float = 0.9,
        num_beams: int = 1,
        dynamic_image_size: bool = True,
        max_tiles_per_image: int = 6,
    ) -> TemporalComparisonResponse:
        """
        Executes a temporal comparison between 2 or more remote-sensing images.

        Args:
            images: List of 2 or more image inputs (paths, URLs, base64, bytes, PIL).
            query: Comparison question/task.
            prompt_style: PromptStyle.EVALUATION, CONVERSATIONAL, or RAW.
            task_token: EarthDial task token (e.g. '[changedet] [hr_rgb_temp_0.5]').
            temperature: Sampling temperature (0.0 = greedy).
            max_new_tokens: Max generated tokens.
            top_p: Nucleus sampling probability.
            num_beams: Beam search beams.
            dynamic_image_size: Whether to tile high-res images dynamically.
            max_tiles_per_image: Maximum dynamic tiles per image.
        """
        if len(images) < 2:
            raise ValueError(f"Temporal comparison requires at least 2 images, but received {len(images)}.")

        t0 = time.time()

        # Load all images into PIL RGB
        pil_images = [self.load_image(img) for img in images]

        # Format prompt
        formatted_prompt = self.format_prompt(
            num_images=len(pil_images),
            query=query,
            style=prompt_style,
            task_token=task_token
        )

        # Execute inference
        response_text, meta_list, device_used = self.model.generate(
            images=pil_images,
            prompt=formatted_prompt,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            num_beams=num_beams,
            dynamic_image_size=dynamic_image_size,
            max_tiles_per_image=max_tiles_per_image,
        )

        elapsed = round(time.time() - t0, 3)

        image_info = [
            ImageResolutionInfo(
                index=m["index"],
                width=m["width"],
                height=m["height"],
                num_tiles=m.get("num_tiles", 1)
            )
            for m in meta_list
        ]

        return TemporalComparisonResponse(
            success=True,
            query=query,
            formatted_prompt=formatted_prompt,
            response=response_text,
            num_images=len(pil_images),
            image_metadata=image_info,
            execution_time_seconds=elapsed,
            device=device_used,
            model_name_or_path=self.model.checkpoint,
            metadata={"temperature": temperature, "max_new_tokens": max_new_tokens}
        )

    def compare_from_request(self, request: TemporalComparisonRequest) -> TemporalComparisonResponse:
        """Helper to invoke compare directly from a Pydantic request object."""
        return self.compare(
            images=request.images,
            query=request.query,
            prompt_style=request.prompt_style,
            task_token=request.task_token,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            num_beams=request.num_beams,
            dynamic_image_size=request.dynamic_image_size,
            max_tiles_per_image=request.max_tiles_per_image,
        )

    @staticmethod
    def get_templates() -> List[PromptTemplateItem]:
        """Returns standard pre-configured templates for change detection."""
        return [
            PromptTemplateItem(
                task_category="General Change Detection",
                name="Semantic Change Identification",
                query="Analyze the images and describe any semantic or structural changes detected between the observations.",
                suggested_tokens="[changedet] [hr_rgb_temp_0.5]",
                description="General purpose remote sensing change detection across two or more acquisitions."
            ),
            PromptTemplateItem(
                task_category="Urbanization & Construction",
                name="Infrastructure & Road Evolution",
                query="Identify newly built roads, residential developments, or commercial structures between the time periods.",
                suggested_tokens="[changedet] [hr_rgb_temp_0.5]",
                description="Detects urban expansion, new buildings, and transportation networks."
            ),
            PromptTemplateItem(
                task_category="Deforestation & Environment",
                name="Vegetation Loss & Land Clearing",
                query="Detect areas where forest canopy or vegetation has been cleared, reduced, or modified.",
                suggested_tokens="[changedet] [hr_rgb_temp_0.5]",
                description="Assesses forest clearing, agriculture conversion, and vegetative health shifts."
            ),
            PromptTemplateItem(
                task_category="Disaster Assessment",
                name="Post-Disaster Structural Damage",
                query="Analyze the pre-disaster and post-disaster images to identify destroyed buildings, flooded zones, or debris fields.",
                suggested_tokens="[changedet] [hr_rgb_temp_0.5]",
                description="Evaluates disaster impact (earthquake, hurricane, flood, wildfire) according to xBD protocols."
            ),
            PromptTemplateItem(
                task_category="Water & Coastal Dynamics",
                name="Water Body & Shoreline Evolution",
                query="Compare the water boundary, reservoir surface area, or coastline changes across the acquisitions.",
                suggested_tokens="[changedet] [hr_rgb_temp_0.5]",
                description="Monitors coastal erosion, lake shrinkage, or flood inundation extent."
            ),
        ]
