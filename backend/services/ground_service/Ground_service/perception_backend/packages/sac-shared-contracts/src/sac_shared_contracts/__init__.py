"""Canonical shared integration contracts for SAC GeoAI Agent.

Source of truth for domain models, tool protocols, and integration interfaces.
"""

__version__ = "0.1.0"

from sac_shared_contracts.common import (
    StrictModel,
    Modality,
    CoordinateFrame,
    PointPx,
    BBoxPx,
    GeoPoint,
    OBB,
    Confidence,
    AssetRef,
    ArtifactRef,
    EvidenceRef,
    Detection,
)
from sac_shared_contracts.tool_protocol import (
    ToolInput,
    ToolOutput,
    ToolCall,
    ToolExecutionResult,
    ToolError,
    ToolErrorCode,
    ToolExecutionStatus,
)
from sac_shared_contracts.perception import (
    ImageAnalyzerInput,
    ImageAnalyzerOutput,
    ObjectDetectorInput,
    ObjectDetectorOutput,
    OBBDetectorInput,
    OBBDetectorOutput,
    SegmentationGroundingInput,
    SegmentationGroundingOutput,
    GroundingResult,
    NMSConfig,
    RotatedNMSConfig,
    TilingConfig,
    GenerationConfig,
)
from sac_shared_contracts.geo_spatial import (
    GeometryRef,
    MeasurementValue,
    GeoMeasurementInput,
    GeoMeasurementOutput,
    SpatialRelationshipInput,
    SpatialRelationship,
    SpatialRelationshipOutput,
    SpatialRelationType,
)
from sac_shared_contracts.roi import (
    ROI,
    ROILocalizerInput,
    ROILocalizerOutput,
)
from sac_shared_contracts.multimodal import (
    ModalityRole,
    ModalityAsset,
    AlignmentReport,
    FusionInput,
    FusionOutput,
)
from sac_shared_contracts.ocr import (
    OCRTextRegion,
    OCRInput,
    OCROutput,
)
from sac_shared_contracts.evidence import (
    Evidence,
    EvidenceType,
)
from sac_shared_contracts.agent import (
    PlanStep,
    AgentPlan,
    Claim,
    FinalResponse,
)

__all__ = [
    # Common
    "StrictModel",
    "Modality",
    "CoordinateFrame",
    "PointPx",
    "BBoxPx",
    "GeoPoint",
    "OBB",
    "Confidence",
    "AssetRef",
    "ArtifactRef",
    "EvidenceRef",
    "Detection",
    # Protocol
    "ToolInput",
    "ToolOutput",
    "ToolCall",
    "ToolExecutionResult",
    "ToolError",
    "ToolErrorCode",
    "ToolExecutionStatus",
    # Perception
    "ImageAnalyzerInput",
    "ImageAnalyzerOutput",
    "ObjectDetectorInput",
    "ObjectDetectorOutput",
    "OBBDetectorInput",
    "OBBDetectorOutput",
    "SegmentationGroundingInput",
    "SegmentationGroundingOutput",
    "GroundingResult",
    "NMSConfig",
    "RotatedNMSConfig",
    "TilingConfig",
    "GenerationConfig",
    # Geo/Spatial
    "GeometryRef",
    "MeasurementValue",
    "GeoMeasurementInput",
    "GeoMeasurementOutput",
    "SpatialRelationshipInput",
    "SpatialRelationship",
    "SpatialRelationshipOutput",
    "SpatialRelationType",
    # ROI
    "ROI",
    "ROILocalizerInput",
    "ROILocalizerOutput",
    # Multimodal
    "ModalityRole",
    "ModalityAsset",
    "AlignmentReport",
    "FusionInput",
    "FusionOutput",
    # OCR
    "OCRTextRegion",
    "OCRInput",
    "OCROutput",
    # Evidence & Agent
    "Evidence",
    "EvidenceType",
    "PlanStep",
    "AgentPlan",
    "Claim",
    "FinalResponse",
]
