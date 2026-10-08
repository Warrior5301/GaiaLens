"""Prompt templates and formatting for EarthMind remote-sensing VLM."""

from __future__ import annotations

from typing import Optional


class EarthMindPrompts:
    """Standardized prompt templates for remote-sensing scene understanding."""

    DEFAULT_CAPTION_PROMPT = (
        "<image>\n"
        "Provide a comprehensive, factual description of this high-resolution remote-sensing image. "
        "Describe visible infrastructure, transportation corridors, natural terrain, and observable objects."
    )

    DEFAULT_SCENE_PROMPT = (
        "<image>\n"
        "Analyze this remote sensing scene. Identify the primary land-use classification, "
        "spatial patterns, and key operational facilities visible in the aerial view."
    )

    @staticmethod
    def format_caption_prompt(custom_instruction: Optional[str] = None) -> str:
        """Format prompt for image captioning."""
        if custom_instruction:
            return f"<image>\n{custom_instruction}"
        return EarthMindPrompts.DEFAULT_CAPTION_PROMPT

    @staticmethod
    def format_vqa_prompt(question: str) -> str:
        """Format prompt for visual question answering."""
        clean_q = question.strip()
        return (
            f"<image>\n"
            f"Question: {clean_q}\n"
            f"Based strictly on visual evidence in the remote sensing image, provide an accurate and direct answer.\n"
            f"Answer:"
        )

    @staticmethod
    def format_scene_prompt() -> str:
        """Format prompt for scene understanding."""
        return EarthMindPrompts.DEFAULT_SCENE_PROMPT
