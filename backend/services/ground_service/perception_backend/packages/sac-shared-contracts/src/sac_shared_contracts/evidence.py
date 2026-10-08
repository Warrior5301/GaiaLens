"""Traceable factual evidence schema linking tool outputs to downstream agent claims."""

from __future__ import annotations

from enum import Enum
from typing import Any
import uuid
from pydantic import Field
from sac_shared_contracts.common import ArtifactRef, StrictModel


class EvidenceType(str, Enum):
    """Functional classification of evidence items."""
    CAPTION = "caption"
    DETECTION = "detection"
    OBB_DETECTION = "obb_detection"
    GROUNDING = "grounding"
    COUNT = "count"
    MEASUREMENT = "measurement"
    SPATIAL_RELATION = "spatial_relation"
    OCR = "ocr"
    FUSION = "fusion"
    QUALITATIVE_OBSERVATION = "qualitative_observation"


class Evidence(StrictModel):
    """Traceable unit of factual evidence produced by deterministic or perception tools."""
    evidence_id: str = Field(
        default_factory=lambda: f"ev_{uuid.uuid4().hex[:8]}",
        description="Unique evidence identifier cited in claims",
    )
    evidence_type: EvidenceType | str = Field(
        description="Functional type of the evidence"
    )
    source_tool: str = Field(
        description="Name of the originating tool (e.g. Object_Detector, Geo_Measurement)"
    )
    source_tool_version: str = Field(
        default="0.1.0", description="Semantic version of the originating tool contract"
    )
    asset_id: str = Field(
        description="Identifier of asset to which this evidence applies"
    )
    payload: dict[str, Any] = Field(
        default_factory=dict, description="Structured factual findings extracted by the tool"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence associated with this piece of evidence"
    )
    artifact_refs: list[ArtifactRef] = Field(
        default_factory=list, description="Derived artifacts supporting this evidence"
    )
