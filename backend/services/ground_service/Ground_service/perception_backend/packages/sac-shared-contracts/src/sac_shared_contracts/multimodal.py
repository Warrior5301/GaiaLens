"""Team C — Multimodal Fusion and Sensor Alignment Contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import Field
from sac_shared_contracts.common import ArtifactRef, AssetRef, StrictModel
from sac_shared_contracts.tool_protocol import ToolInput, ToolOutput


class ModalityRole(str, Enum):
    """Functional role of an asset within a multimodal fusion process."""
    OPTICAL = "optical"
    SAR = "sar"
    INFRARED = "infrared"
    THERMAL = "thermal"
    REFERENCE = "reference"


class ModalityAsset(StrictModel):
    """Specification of an asset paired with its sensor fusion role."""
    asset: AssetRef = Field(description="Remote sensing asset reference")
    role: ModalityRole = Field(description="Sensor role within the fusion operation")


class AlignmentReport(StrictModel):
    """Diagnostic report documenting spatial coregistration accuracy."""
    registered: bool = Field(description="Whether coregistration succeeded within tolerance")
    residual_rmse_px: float = Field(
        ge=0.0, description="Root Mean Square Error of alignment residuals in pixels"
    )
    target_crs: str = Field(description="Common output coordinate reference system (e.g. EPSG:3857)")
    target_resolution_m: float = Field(
        gt=0.0, description="Common output ground sample distance in meters"
    )
    notes: Optional[str] = Field(default=None, description="Alignment diagnostic observations or caveats")


class FusionInput(ToolInput):
    """Input contract for SAR_Optical_Fuser / Multimodal fusion tools."""
    reference_asset: AssetRef = Field(description="Anchor image providing spatial geometry reference")
    secondary_assets: list[ModalityAsset] = Field(
        description="List of secondary modality assets to align and fuse"
    )
    target_crs: Optional[str] = Field(default=None, description="Optional target CRS override")
    target_resolution_m: Optional[float] = Field(
        default=None, gt=0.0, description="Target output spatial resolution in meters"
    )


class FusionOutput(ToolOutput):
    """Output contract from multimodal fusion operations."""
    fused_artifact: ArtifactRef = Field(
        description="Artifact reference to derived fused composite or aligned multi-band tensor"
    )
    alignment_report: AlignmentReport = Field(
        description="Formal coregistration and alignment quality metrics"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Fusion algorithm metadata")
