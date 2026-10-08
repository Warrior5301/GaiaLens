"""Numeric question capability routing for Object_Detector.

Determines whether a question can be answered by the discrete object detector
versus requiring VLM scene understanding or spatial/continuous metrics.
"""

from __future__ import annotations

import re
from typing import Optional
from pydantic import BaseModel, ConfigDict, Field
from app.tools.object_detector.class_mapping import ClassMapper


class CapabilityAssessment(BaseModel):
    """Structured assessment evaluating if Object_Detector can answer a question."""
    model_config = ConfigDict(extra="forbid")

    target_class: Optional[str] = Field(default=None, description="Physical detector target class if supported")
    detector_supported: bool = Field(description="Whether the detector can directly or with ROI answer this")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in routing assessment")
    reason: str = Field(description="Human-readable routing rationale")
    requires_roi: bool = Field(default=False, description="Whether question references a sub-region or spatial anchor")
    roi_query: Optional[str] = Field(default=None, description="Extracted regional anchor (e.g. 'near the harbor')")


def can_detector_answer(question: str) -> CapabilityAssessment:
    """Analyze a natural language question to determine if Object_Detector can answer it.

    Examples:
    - "How many airplanes are there?" -> supported (airplane)
    - "How many ships are near the harbor?" -> supported after ROI localization
    - "What is the average height of buildings?" -> not supported
    - "How many distinct land-cover types are visible?" -> not supported
    """
    clean_q = question.strip()

    # 1. Check for numerical / counting intent
    counting_match = re.search(r"\b(how\s+many|count|number\s+of|total\s+number)\b", clean_q, re.I)
    is_counting = bool(counting_match)

    # 2. Check explicitly unsupported patterns (height, area, land-cover, etc.)
    unsupported_check = ClassMapper.map_phrase(clean_q)
    if not unsupported_check.is_supported and unsupported_check.notes and "not a 2D detector capability" in unsupported_check.notes:
        return CapabilityAssessment(
            target_class=None,
            detector_supported=False,
            confidence=0.98,
            reason=unsupported_check.notes,
        )

    if not is_counting:
        return CapabilityAssessment(
            target_class=None,
            detector_supported=False,
            confidence=0.92,
            reason="Question does not request an object count or numerical enumeration",
        )

    # 3. Check for regional or contextual sub-scope (e.g. "near the harbor", "in the parking lot")
    roi_pattern = re.search(r"\b(near|in|around|inside|at|along)\s+(the\s+)?([a-zA-Z\s]+?)(?:\?|$)", clean_q, re.I)
    requires_roi = False
    roi_query: Optional[str] = None
    if roi_pattern:
        candidate_anchor = roi_pattern.group(3).strip()
        if candidate_anchor and candidate_anchor not in {"there", "image", "scene"}:
            requires_roi = True
            roi_query = candidate_anchor

    # 4. Extract target object noun phrase
    # Remove counting prefix
    core_text = re.sub(r"^(how\s+many|what\s+is\s+the\s+number\s+of|count\s+the)\s+", "", clean_q, flags=re.I)
    core_text = re.sub(r"\s+(are\s+there|visible|present|in\s+the\s+image).*", "", core_text, flags=re.I)

    mapping = ClassMapper.map_phrase(core_text)

    if mapping.is_supported and mapping.detector_class:
        if requires_roi:
            return CapabilityAssessment(
                target_class=mapping.detector_class,
                detector_supported=True,
                confidence=mapping.confidence,
                reason=f"Supported for counting '{mapping.detector_class}' after ROI localization on '{roi_query}'",
                requires_roi=True,
                roi_query=roi_query,
            )
        return CapabilityAssessment(
            target_class=mapping.detector_class,
            detector_supported=True,
            confidence=mapping.confidence,
            reason=f"Supported closed-set counting query for class '{mapping.detector_class}'",
            requires_roi=False,
            roi_query=None,
        )

    return CapabilityAssessment(
        target_class=None,
        detector_supported=False,
        confidence=0.88,
        reason=f"Target entity in '{clean_q}' is not a recognized discrete detector category",
        requires_roi=requires_roi,
        roi_query=roi_query,
    )
