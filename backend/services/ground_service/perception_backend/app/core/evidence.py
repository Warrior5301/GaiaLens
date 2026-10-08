"""Evidence creation and builder utilities for perception outputs."""

from __future__ import annotations

from typing import Any
from app.schemas.evidence import Evidence, EvidenceType
from app.schemas.perception import ImageAnalyzerOutput


class EvidenceBuilder:
    """Builds standard traceable Evidence records from perception outputs."""

    @staticmethod
    def from_image_analyzer(
        output: ImageAnalyzerOutput,
        asset_id: str,
        source_tool: str = "Image_Analyzer",
        source_tool_version: str = "1.0.0",
    ) -> list[Evidence]:
        """Convert ImageAnalyzerOutput into structured Evidence objects."""
        evidences: list[Evidence] = []
        conf_score = output.confidence.score if output.confidence else 0.85

        if output.caption:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.CAPTION,
                    source_tool=source_tool,
                    source_tool_version=source_tool_version,
                    asset_id=asset_id,
                    payload={
                        "task_type": output.task_type,
                        "caption": output.caption,
                        "observations": output.observations,
                        "model": output.model_metadata.get("architecture", "EarthMind"),
                    },
                    confidence=conf_score,
                )
            )

        if output.answer:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.QUALITATIVE_OBSERVATION,
                    source_tool=source_tool,
                    source_tool_version=source_tool_version,
                    asset_id=asset_id,
                    payload={
                        "task_type": output.task_type,
                        "answer": output.answer,
                        "observations": output.observations,
                        "model": output.model_metadata.get("architecture", "EarthMind"),
                    },
                    confidence=conf_score,
                )
            )

        # If only scene_understanding observations were produced
        if not output.caption and not output.answer and output.observations:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.QUALITATIVE_OBSERVATION,
                    source_tool=source_tool,
                    source_tool_version=source_tool_version,
                    asset_id=asset_id,
                    payload={
                        "task_type": output.task_type,
                        "observations": output.observations,
                    },
                    confidence=conf_score,
                )
            )

        return evidences
