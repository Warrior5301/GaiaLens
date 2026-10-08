"""Domain objects and foundational schemas for SAC GeoAI Agent.

Canonical definitions are imported directly from sac_shared_contracts to ensure
a single source of truth across all workstreams and tools.
"""

from __future__ import annotations

from typing import Any, Optional
from pydantic import Field

from sac_shared_contracts.common import (
    ArtifactRef,
    ArtifactType,
    AssetRef,
    BBoxPx,
    Confidence,
    CoordinateFrame,
    Detection,
    EvidenceRef,
    GeoPoint,
    Modality,
    OBB,
    PointPx,
    StrictModel,
)
from sac_shared_contracts.roi import ROI


class MaskArtifact(StrictModel):
    """Segmentation mask metadata referencing an external binary/multiclass mask artifact."""
    mask_id: str = Field(description="Unique identifier for the mask")
    source_asset_id: str = Field(description="Originating asset ID")
    object_id: Optional[str] = Field(
        default=None, description="Associated object or instance ID if applicable"
    )
    mask_asset_ref: ArtifactRef = Field(description="Artifact reference to stored mask file")
    polygon: Optional[list[list[float]]] = Field(
        default=None, description="Contour polygon coordinates [[x1, y1], [x2, y2], ...]"
    )
    area_pixels: Optional[float] = Field(
        default=None, ge=0.0, description="Total segmented pixel count/area"
    )
    confidence: Optional[Confidence] = Field(
        default=None, description="Segmentation confidence"
    )


__all__ = [
    "StrictModel",
    "CoordinateFrame",
    "PointPx",
    "BBoxPx",
    "GeoPoint",
    "OBB",
    "Confidence",
    "Modality",
    "AssetRef",
    "ArtifactType",
    "ArtifactRef",
    "EvidenceRef",
    "Detection",
    "ROI",
    "MaskArtifact",
]
