"""Image_Analyzer tool implementation backed by EarthMind VLM.

Flow:
    AssetRef
      -> VisionRuntime
      -> ImageAnalyzerProcessor
      -> EarthMindBackend
      -> ImageAnalyzerOutputParser
      -> ImageAnalyzerOutput
      -> Schema Validation
      -> Evidence Generation
"""

from __future__ import annotations

import os
from typing import Any, Optional
from app.core.evidence import EvidenceBuilder
from app.core.logging import logger
from app.core.vision_runtime import VisionRuntime
from app.models.base import ModelBackend
from app.models.loaders import ModelLoader
from app.schemas.evidence import Evidence
from app.schemas.perception import ImageAnalyzerInput, ImageAnalyzerOutput
from app.tools.base import BaseTool
from app.tools.image_analyzer.model import ImageAnalyzerModelInterface
from app.tools.image_analyzer.parser import ImageAnalyzerOutputParser
from app.tools.image_analyzer.processor import ImageAnalyzerProcessor


class ImageAnalyzer(BaseTool[ImageAnalyzerInput, ImageAnalyzerOutput]):
    """Production-grade Semantic Vision-Language Tool using EarthMind VLM."""

    name = "Image_Analyzer"
    version = "1.0.0"
    description = "Semantic remote-sensing VLM for captioning, qualitative VQA, and scene understanding"
    input_schema_class = ImageAnalyzerInput
    output_schema_class = ImageAnalyzerOutput

    def __init__(
        self,
        runtime: Optional[VisionRuntime] = None,
        model_interface: Optional[ImageAnalyzerModelInterface] = None,
        model_backend: Optional[ModelBackend] = None,
        model_id: str | None = None,
        model_loader: Optional[ModelLoader] = None,
    ) -> None:
        self.runtime = runtime or VisionRuntime()
        selected_model_id = model_id or os.getenv(
            "SAC_IMAGE_ANALYZER_MODEL_ID", "earthmind_v1"
        )
        if model_interface is not None:
            self.model_interface = model_interface
        else:
            self.model_interface = ImageAnalyzerModelInterface(
                model_id=selected_model_id,
                backend=model_backend,
                model_loader=model_loader,
            )

    def execute(self, input_data: ImageAnalyzerInput) -> ImageAnalyzerOutput:
        """Execute semantic analysis task through complete modular pipeline."""
        logger.info(
            f"[{self.name}] Executing task_type: '{input_data.task_type}' on asset: '{input_data.asset.asset_id}'"
        )

        # 1. Image & Prompt Preprocessing
        prepared_img, formatted_prompt = ImageAnalyzerProcessor.process(
            input_data=input_data,
            runtime=self.runtime,
            target_size=(512, 512),
        )

        # 2. Model Inference via EarthMind Backend
        raw_result = self.model_interface.generate(
            image=prepared_img,
            prompt=formatted_prompt,
            task_type=input_data.task_type,
            generation=input_data.generation,
        )

        raw_text = raw_result.get("raw_text", "")
        confidence_score = float(raw_result.get("confidence_score", 0.85))
        model_meta = raw_result.get("metadata", {})

        # 3. Output Parsing & Structuring
        structured_output = ImageAnalyzerOutputParser.parse(
            raw_text=raw_text,
            confidence_score=confidence_score,
            task_type=input_data.task_type,
            model_metadata=model_meta,
            question=input_data.question,
        )

        return structured_output

    def to_evidence(self, output: ImageAnalyzerOutput, asset_id: str) -> list[Evidence]:
        """Convert validated output into traceable Evidence objects."""
        return EvidenceBuilder.from_image_analyzer(
            output=output,
            asset_id=asset_id,
            source_tool=self.name,
            source_tool_version=self.version,
        )
