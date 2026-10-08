"""Parser module transforming raw model text into structured ImageAnalyzerOutput."""

from __future__ import annotations

import re
from typing import Any, Optional
from app.schemas.common import Confidence
from app.schemas.perception import ImageAnalyzerOutput, ImageTaskType


class ImageAnalyzerOutputParser:
    """Parses and normalizes raw text generations from EarthMind into structured contracts."""

    @staticmethod
    def normalize_vqa_answer(raw_text: str, question: Optional[str] = None) -> str:
        """Clean and normalize textual answer for benchmark scoring."""
        text = raw_text.strip()
        # Remove common chat prefixes like 'Answer:', 'The answer is'
        text = re.sub(r"^(answer:\s*|the answer is\s*)", "", text, flags=re.IGNORECASE).strip()

        # Binary normalization
        clean_lower = text.lower()
        if question and any(question.lower().startswith(w) for w in ["is ", "are ", "do ", "does ", "can "]):
            clean_words = re.sub(r"[^\w\s]", "", clean_lower).split()
            if "yes" in clean_words:
                return "yes"
            if "no" in clean_words:
                return "no"

        return text

    @staticmethod
    def extract_observations(raw_text: str) -> list[str]:
        """Extract individual factual observation sentences from text."""
        sentences = re.split(r"[.!?]\s+", raw_text.strip())
        obs = [s.strip() for s in sentences if len(s.strip()) > 10]
        return obs[:5]  # Cap at top 5 observations

    @classmethod
    def parse(
        cls,
        raw_text: str,
        confidence_score: float,
        task_type: ImageTaskType,
        model_metadata: dict[str, Any],
        question: Optional[str] = None,
    ) -> ImageAnalyzerOutput:
        """Parse raw model outputs into a verified ImageAnalyzerOutput model."""
        observations = cls.extract_observations(raw_text)
        model_source = str(model_metadata.get("model_id", "image_analyzer"))
        task_name = getattr(task_type, "value", task_type)
        conf = Confidence(
            score=max(0.0, min(1.0, confidence_score)),
            calibrated=False,
            source=f"{model_source}:{task_name}",
        )

        caption: Optional[str] = None
        answer: Optional[str] = None

        if task_type == "caption":
            caption = raw_text.strip()
        elif task_type == "vqa":
            answer = cls.normalize_vqa_answer(raw_text, question)
        else:  # scene_understanding
            caption = raw_text.strip()
            answer = None

        return ImageAnalyzerOutput(
            task_type=task_type,
            caption=caption,
            answer=answer,
            observations=observations,
            confidence=conf,
            raw_text=raw_text,
            model_metadata=model_metadata,
        )
