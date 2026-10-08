"""Team B — Geospatial Measurement and Spatial Reasoning Contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import Field, model_validator
from sac_shared_contracts.common import (
    AssetRef,
    BBoxPx,
    Confidence,
    GeoPoint,
    PointPx,
    StrictModel,
)
from sac_shared_contracts.tool_protocol import ToolInput, ToolOutput

GeometryType = Literal["bbox_px", "polygon", "point", "artifact"]


class GeometryRef(StrictModel):
    """Discriminated spatial geometry reference with semantic validation."""
    geometry_type: GeometryType = Field(description="Active discriminator type for geometry representation")
    bbox_px: Optional[BBoxPx] = Field(default=None, description="Axis-aligned bounding box")
    polygon: Optional[list[list[float]]] = Field(
        default=None, description="Contour polygon coordinates [[x1, y1], [x2, y2], ...]"
    )
    point: Optional[PointPx | GeoPoint] = Field(default=None, description="2D pixel point or geographic point")
    artifact_id: Optional[str] = Field(default=None, description="Derived artifact identifier (e.g. mask or shapefile)")

    @model_validator(mode="after")
    def validate_discriminated_geometry(self) -> GeometryRef:
        gt = self.geometry_type
        if gt == "bbox_px" and self.bbox_px is None:
            raise ValueError("geometry_type='bbox_px' requires 'bbox_px' field to be set")
        elif gt == "polygon":
            if self.polygon is None or len(self.polygon) < 3:
                raise ValueError("geometry_type='polygon' requires 'polygon' with at least 3 vertices")
        elif gt == "point" and self.point is None:
            raise ValueError("geometry_type='point' requires 'point' field to be set")
        elif gt == "artifact" and not self.artifact_id:
            raise ValueError("geometry_type='artifact' requires 'artifact_id' field to be set")
        return self


class MeasurementValue(StrictModel):
    """Quantitative measurement with unit and confidence."""
    value: float = Field(description="Numerical scalar value")
    unit: str = Field(description="Unit of measurement: e.g. 'm', 'm2', 'km', 'km2', 'hectares', 'degrees'")
    confidence: Confidence = Field(description="Measurement confidence and method")
    method: str = Field(default="pixel_gsd", description="Computation method: e.g. geodesic, projected, gsd")


class GeoMeasurementInput(ToolInput):
    """Input contract for Geo_Measurement tool."""
    asset: AssetRef = Field(description="Reference asset containing geospatial scale and CRS")
    geometry: GeometryRef = Field(description="Target geometry to measure")
    measurement_types: list[str] = Field(
        default_factory=lambda: ["area"],
        description="Types of measurement requested: e.g. area, perimeter, length, orientation",
    )
    crs_override: Optional[str] = Field(
        default=None, description="Optional CRS override for projected calculations"
    )


class GeoMeasurementOutput(ToolOutput):
    """Output contract from Geo_Measurement tool."""
    measurements: dict[str, MeasurementValue] = Field(
        description="Map of measurement_type to quantitative MeasurementValue"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Execution metadata")


class SpatialRelationType(str, Enum):
    """Topological, directional, and metric spatial relationship types."""
    CONTAINS = "contains"
    WITHIN = "within"
    INTERSECTS = "intersects"
    DISJOINT = "disjoint"
    ADJACENT_TO = "adjacent_to"
    NORTH_OF = "north_of"
    SOUTH_OF = "south_of"
    EAST_OF = "east_of"
    WEST_OF = "west_of"
    DISTANCE = "distance"


class SpatialRelationship(StrictModel):
    """Evaluated spatial relationship between two geometries."""
    relation_type: SpatialRelationType = Field(description="Target spatial relation")
    satisfied: bool = Field(description="Whether the relation holds true between subject and object")
    confidence: Confidence = Field(description="Confidence in relationship assessment")
    distance_m: Optional[float] = Field(default=None, ge=0.0, description="Metric distance between entities if applicable")
    details: dict[str, Any] = Field(default_factory=dict, description="Diagnostic details")


class SpatialRelationshipInput(ToolInput):
    """Input contract for Spatial_Relationship tool."""
    asset: AssetRef = Field(description="Reference asset")
    subject: GeometryRef = Field(description="Primary subject geometry")
    object_ref: GeometryRef = Field(description="Secondary reference geometry")
    relation_types: list[SpatialRelationType] = Field(
        description="List of relationship hypotheses to evaluate"
    )
    distance_threshold_m: Optional[float] = Field(
        default=None, ge=0.0, description="Optional distance threshold for adjacency/proximity"
    )


class SpatialRelationshipOutput(ToolOutput):
    """Output contract from Spatial_Relationship tool."""
    relationships: list[SpatialRelationship] = Field(
        description="Evaluated outcomes for each requested relationship"
    )
    metadata: dict[str, Any] = Field(default_factory=dict, description="Calculation metadata")
