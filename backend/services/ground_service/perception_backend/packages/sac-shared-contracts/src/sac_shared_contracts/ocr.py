"""Team C — Optical Character Recognition (OCR) Contracts for Remote Sensing Imagery."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import Field
from sac_shared_contracts.common import AssetRef, BBoxPx, Confidence, StrictModel
from sac_shared_contracts.tool_protocol import ToolInput, ToolOutput


class OCRTextRegion(StrictModel):
    """Recognized textual transcription within a localized bounding box."""
    text_id: str = Field(description="Unique text instance identifier")
    text: str = Field(description="Extracted textual transcription")
    bbox_px: BBoxPx = Field(description="Bounding box of text region in canonical pixel coordinates")
    confidence: Confidence = Field(description="Transcription confidence score")
    language: Optional[str] = Field(default=None, description="Detected or specified language code (e.g. 'en')")
    rotation_deg: Optional[float] = Field(default=None, description="Text baseline orientation in degrees")


class OCRInput(ToolInput):
    """Input contract for OCR_Extractor tool."""
    asset: AssetRef = Field(description="Target image asset reference")
    roi: Optional[BBoxPx] = Field(default=None, description="Optional region constraint for text extraction")
    languages: list[str] = Field(
        default_factory=lambda: ["en"], description="Target language codes for recognition"
    )


class OCROutput(ToolOutput):
    """Output contract from OCR_Extractor tool."""
    regions: list[OCRTextRegion] = Field(
        default_factory=list, description="Localized textual detections"
    )
    full_text: str = Field(description="Concatenated reading-order text")
    metadata: dict[str, Any] = Field(default_factory=dict, description="OCR engine metadata")
