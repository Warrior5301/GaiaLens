"""Canonical domain models and spatial foundation contracts.

COORDINATE CONVENTIONS (CANONICAL):
- Pixel Coordinates:
    * Origin (0, 0): Top-Left corner of image
    * X-axis: Points rightwards (columns)
    * Y-axis: Points downwards (rows)
- Bounding Box:
    * x_min, y_min, x_max, y_max
    * Constraint: x_max >= x_min and y_max >= y_min
- Oriented Bounding Box (OBB):
    * cx, cy, width, height, angle_deg
    * Canonical angle convention: 'ccw_from_image_x' (Counter-clockwise rotation in degrees from positive image X-axis)
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal, Optional
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Immutable base model enforcing strict schema boundaries (no extra fields allowed)."""
    model_config = ConfigDict(
        extra="forbid",
        validate_assignment=True,
        populate_by_name=True,
        use_enum_values=True,
    )


class Modality(str, Enum):
    """Supported sensor modalities for input imagery."""
    OPTICAL = "optical"
    SAR = "sar"
    INFRARED = "infrared"
    THERMAL = "thermal"
    MULTISPECTRAL = "multispectral"
    HYPERSPECTRAL = "hyperspectral"
    PANCHROMATIC = "panchromatic"


class CoordinateFrame(str, Enum):
    """Coordinate reference frames supported across SAC GeoAI."""
    PIXEL = "pixel"
    NORMALIZED = "normalized"
    GEOGRAPHIC = "geographic"
    PROJECTED = "projected"


class PointPx(StrictModel):
    """2D point in canonical image pixel coordinates (origin: top-left, x: right, y: down)."""
    x: float = Field(description="Horizontal pixel coordinate (columns from left edge)")
    y: float = Field(description="Vertical pixel coordinate (rows from top edge)")


class BBoxPx(StrictModel):
    """Canonical 2D Axis-Aligned Bounding Box in image pixel coordinates."""
    x_min: float = Field(description="Left pixel coordinate")
    y_min: float = Field(description="Top pixel coordinate")
    x_max: float = Field(description="Right pixel coordinate")
    y_max: float = Field(description="Bottom pixel coordinate")

    @model_validator(mode="after")
    def validate_bounds(self) -> BBoxPx:
        if self.x_max < self.x_min:
            raise ValueError(
                f"Invalid bounding box: x_max ({self.x_max}) must be >= x_min ({self.x_min})"
            )
        if self.y_max < self.y_min:
            raise ValueError(
                f"Invalid bounding box: y_max ({self.y_max}) must be >= y_min ({self.y_min})"
            )
        return self

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> PointPx:
        return PointPx(x=(self.x_min + self.x_max) / 2.0, y=(self.y_min + self.y_max) / 2.0)

    def to_xyxy(self) -> list[float]:
        return [self.x_min, self.y_min, self.x_max, self.y_max]


class GeoPoint(StrictModel):
    """Geographic coordinate representation."""
    latitude: float = Field(ge=-90.0, le=90.0, description="Latitude in decimal degrees")
    longitude: float = Field(ge=-180.0, le=180.0, description="Longitude in decimal degrees")
    altitude_m: Optional[float] = Field(default=None, description="Optional altitude above sea level in meters")
    crs: str = Field(default="EPSG:4326", description="Coordinate Reference System identifier")


class OBB(StrictModel):
    """Canonical Oriented Bounding Box representation."""
    cx: float = Field(description="Center X pixel coordinate")
    cy: float = Field(description="Center Y pixel coordinate")
    width: float = Field(ge=0.0, description="Width along major box axis")
    height: float = Field(ge=0.0, description="Height along minor box axis")
    angle_deg: float = Field(description="Rotation angle in degrees")
    angle_convention: str = Field(
        default="ccw_from_image_x",
        description="Canonical angle convention: Counter-Clockwise from image positive X-axis",
    )


class Confidence(StrictModel):
    """Quantitative confidence score and calibration details."""
    score: float = Field(ge=0.0, le=1.0, description="Confidence score between 0.0 and 1.0")
    calibrated: bool = Field(
        default=False, description="Whether the score represents a calibrated probability"
    )
    source: str = Field(
        description="Origin of confidence estimate: e.g. model_logits, ensemble_vote, rule_heuristic"
    )


class AssetRef(StrictModel):
    """Reference to an input satellite/aerial imagery asset (does NOT store image tensors)."""
    asset_id: str = Field(description="Unique asset identifier")
    uri: str = Field(description="Storage URI or filesystem path to image asset")
    modality: Modality = Field(
        default=Modality.OPTICAL, description="Sensor modality classification"
    )
    width_px: int = Field(gt=0, description="Image width in pixels")
    height_px: int = Field(gt=0, description="Image height in pixels")
    gsd_m: Optional[float] = Field(
        default=None, gt=0.0, description="Ground Sample Distance in meters per pixel"
    )
    crs: Optional[str] = Field(
        default=None, description="Coordinate Reference System (e.g. EPSG:4326, EPSG:3857)"
    )
    acquisition_time: Optional[str] = Field(
        default=None, description="ISO-8601 acquisition timestamp"
    )
    bands: Optional[list[str]] = Field(
        default=None, description="List of spectral band identifiers (e.g. ['R', 'G', 'B', 'NIR'])"
    )


ArtifactType = Literal[
    "image",
    "crop",
    "roi",
    "detection_set",
    "obb_set",
    "segmentation_mask",
    "polygon",
    "measurement",
    "spatial_relation",
    "ocr_result",
    "aligned_asset",
    "fusion_feature",
    "change_map",
]


class ArtifactRef(StrictModel):
    """Reference to derived data artifacts stored externally to keep agent state lightweight."""
    artifact_id: str = Field(description="Unique derived artifact identifier")
    artifact_type: ArtifactType = Field(description="Class of artifact stored")
    uri: str = Field(description="Storage URI or relative artifact path")
    source_asset_ids: list[str] = Field(
        default_factory=list, description="IDs of source assets that contributed to this artifact"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary artifact metadata properties"
    )

    @model_validator(mode="before")
    @classmethod
    def handle_legacy_source_asset_id(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "source_asset_id" in data and "source_asset_ids" not in data:
                val = data.pop("source_asset_id")
                data["source_asset_ids"] = [val] if val else []
        return data


class EvidenceRef(StrictModel):
    """Reference pointer to a factual evidence record."""
    evidence_id: str = Field(description="Unique evidence identifier")
    source_tool: str = Field(description="Tool that produced this evidence item")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence associated with this evidence")
    uri: Optional[str] = Field(default=None, description="Optional URI to underlying evidence payload")


class Detection(StrictModel):
    """Canonical single object detection instance."""
    object_id: str = Field(description="Unique object instance identifier")
    class_name: str = Field(description="Predicted semantic object class")
    confidence: Confidence = Field(description="Detection confidence details")
    bbox_px: BBoxPx = Field(description="Axis-Aligned Bounding Box in pixel coordinates")
    obb: Optional[OBB] = Field(
        default=None, description="Oriented Bounding Box if orientation was detected"
    )
    center_px: Optional[PointPx] = Field(
        default=None, description="Centroid coordinate of detection in pixels"
    )
    geolocation: Optional[GeoPoint] = Field(
        default=None, description="Geographic coordinates if CRS and georeference are known"
    )

    @model_validator(mode="after")
    def ensure_center(self) -> Detection:
        if self.center_px is None:
            self.center_px = self.bbox_px.center
        return self
