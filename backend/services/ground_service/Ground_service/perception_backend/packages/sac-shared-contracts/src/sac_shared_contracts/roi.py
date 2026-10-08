"""Team C — Region of Interest (ROI) Contracts."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import Field
from sac_shared_contracts.common import (
    AssetRef,
    BBoxPx,
    Confidence,
    CoordinateFrame,
    PointPx,
    StrictModel,
)
from sac_shared_contracts.tool_protocol import ToolInput, ToolOutput


class ROI(StrictModel):
    """Canonical Region of Interest defining a focused spatial scope within an asset."""
    roi_id: str = Field(description="Unique ROI identifier")
    source_asset_id: str = Field(description="Originating asset ID")
    bbox_px: BBoxPx = Field(description="Axis-Aligned Bounding Box in canonical pixel frame")
    polygon: Optional[list[PointPx]] = Field(
        default=None, description="Optional finer polygon boundary of ROI"
    )
    reason: Optional[str] = Field(
        default=None, description="Semantic rationale for ROI localization (e.g. prompt, prior detection)"
    )
    confidence: Optional[Confidence] = Field(
        default=None, description="Confidence in ROI localization accuracy"
    )
    coordinate_frame: CoordinateFrame = Field(
        default=CoordinateFrame.PIXEL, description="Coordinate system convention (canonical: pixel)"
    )


class ROILocalizerInput(ToolInput):
    """Input contract for ROI_Localizer tool."""
    asset: AssetRef = Field(description="Target image asset reference")
    target_description: str = Field(
        description="Natural language expression describing region of interest (e.g. 'harbor area')"
    )
    max_rois: int = Field(default=5, gt=0, description="Maximum number of candidate ROIs to return")
    min_confidence: float = Field(default=0.3, ge=0.0, le=1.0, description="Minimum confidence cutoff")


class ROILocalizerOutput(ToolOutput):
    """Output contract from ROI_Localizer tool."""
    rois: list[ROI] = Field(default_factory=list, description="Extracted localized Regions of Interest")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Localizer metadata")
