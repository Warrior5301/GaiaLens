"""Team A — Perception and Vision-Language Tool Contracts.

Covers input and output interfaces for:
- Image_Analyzer
- Object_Detector
- OBB_Detector
- Segmentation_Grounding
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from pydantic import Field, model_validator
from sac_shared_contracts.common import (
    ArtifactRef,
    AssetRef,
    BBoxPx,
    Confidence,
    Detection,
    OBB,
    StrictModel,
)
from sac_shared_contracts.roi import ROI
from sac_shared_contracts.tool_protocol import ToolInput, ToolOutput


class GenerationConfig(StrictModel):
    """Text generation parameters for VLM reasoning."""
    max_new_tokens: int = Field(default=512, gt=0, description="Maximum new token generation limit")
    temperature: float = Field(default=0.2, ge=0.0, le=2.0, description="Sampling temperature")
    do_sample: bool = Field(default=False, description="Whether to sample or use greedy decoding")


class NMSConfig(StrictModel):
    """Non-Maximum Suppression configuration for axis-aligned bounding boxes."""
    iou_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="IoU threshold for suppression")
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence score cutoff")
    max_detections: int = Field(default=300, gt=0, description="Maximum detections to retain")


class RotatedNMSConfig(StrictModel):
    """Non-Maximum Suppression configuration for oriented bounding boxes."""
    iou_threshold: float = Field(default=0.4, ge=0.0, le=1.0, description="Rotated IoU cutoff")
    score_threshold: float = Field(default=0.25, ge=0.0, le=1.0, description="Confidence cutoff")
    max_detections: int = Field(default=300, gt=0, description="Maximum detections to retain")


class TilingConfig(StrictModel):
    """Configuration for sliding-window patch tiling over large satellite imagery."""
    enabled: bool = Field(default=False, description="Whether to enable sliding window tiling")
    tile_size_px: int = Field(default=512, gt=0, description="Square tile dimension in pixels")
    overlap_px: int = Field(default=64, ge=0, description="Pixel overlap between adjacent tiles")


# ============================================================================
# 1. Image_Analyzer Contracts
# ============================================================================

ImageTaskType = Literal["caption", "vqa", "scene_understanding"]


class ImageAnalyzerInput(ToolInput):
    """Input contract for Image_Analyzer semantic VLM tool."""
    asset: AssetRef = Field(description="Target image asset reference")
    task_type: ImageTaskType = Field(description="Task mode: caption, vqa, or scene_understanding")
    question: Optional[str] = Field(
        default=None, description="Natural language question (mandatory for vqa task)"
    )
    roi: Optional[BBoxPx | ROI] = Field(
        default=None, description="Optional bounding box or ROI spatial sub-window"
    )
    generation: GenerationConfig = Field(
        default_factory=GenerationConfig, description="Text generation parameters"
    )

    @model_validator(mode="after")
    def validate_vqa_question(self) -> ImageAnalyzerInput:
        if self.task_type == "vqa" and not self.question:
            raise ValueError("question must be provided when task_type is 'vqa'")
        return self


class ImageAnalyzerOutput(ToolOutput):
    """Output contract from Image_Analyzer tool."""
    task_type: ImageTaskType = Field(description="Task executed")
    answer: Optional[str] = Field(default=None, description="Normalized answer for VQA")
    caption: Optional[str] = Field(default=None, description="Descriptive caption")
    observations: list[str] = Field(
        default_factory=list, description="Extracted qualitative scene observations"
    )
    confidence: Optional[Confidence] = Field(default=None, description="Confidence assessment")
    raw_text: Optional[str] = Field(default=None, description="Raw model generation text")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Model metadata")


# ============================================================================
# 2. Object_Detector Contracts
# ============================================================================

class ObjectDetectorInput(ToolInput):
    """Input contract for closed-set Object_Detector."""
    asset: AssetRef = Field(description="Target image asset reference")
    target_classes: list[str] = Field(
        default_factory=list, description="Classes to detect; empty list indicates all known classes"
    )
    roi: Optional[BBoxPx | ROI] = Field(
        default=None, description="Optional spatial ROI sub-window (BBoxPx or ROI instance)"
    )
    confidence_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Minimum detection confidence score"
    )
    nms_config: NMSConfig = Field(
        default_factory=NMSConfig, description="NMS suppression parameters"
    )
    tiling_config: TilingConfig = Field(
        default_factory=TilingConfig, description="Tiling configuration for high-resolution images"
    )


class ObjectDetectorOutput(ToolOutput):
    """Output contract from Object_Detector."""
    detections: list[Detection] = Field(
        default_factory=list, description="List of localized object detections"
    )
    count: int = Field(ge=0, description="Exact detection count: len(valid_detections)")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Detector metadata")
    processing_metrics: dict[str, Any] = Field(default_factory=dict, description="Inference timing metrics")

    @model_validator(mode="after")
    def validate_count_consistency(self) -> ObjectDetectorOutput:
        if self.count != len(self.detections):
            raise ValueError(
                f"Detection count mismatch: count={self.count} but len(detections)={len(self.detections)}"
            )
        return self


# ============================================================================
# 3. OBB_Detector Contracts
# ============================================================================

class OBBDetectorInput(ToolInput):
    """Input contract for Oriented Bounding Box detector."""
    asset: AssetRef = Field(description="Target image asset reference")
    target_classes: list[str] = Field(
        default_factory=list, description="Target object classes to detect"
    )
    roi: Optional[BBoxPx] = Field(default=None, description="Optional spatial ROI sub-window")
    confidence_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Minimum detection confidence cutoff"
    )
    rotated_nms_config: RotatedNMSConfig = Field(
        default_factory=RotatedNMSConfig, description="Rotated NMS parameters"
    )
    tiling_config: TilingConfig = Field(
        default_factory=TilingConfig, description="Tiling configuration"
    )


class OBBDetectorOutput(ToolOutput):
    """Output contract from OBB_Detector."""
    detections: list[Detection] = Field(
        default_factory=list, description="List of detections with orientation (OBB) populated"
    )
    count: int = Field(ge=0, description="Exact detection cardinality: len(valid_detections)")
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Model metadata")
    processing_metrics: dict[str, Any] = Field(default_factory=dict, description="Inference metrics")

    @model_validator(mode="after")
    def validate_obb_detections(self) -> OBBDetectorOutput:
        if self.count != len(self.detections):
            raise ValueError(
                f"OBB count mismatch: count={self.count} but len(detections)={len(self.detections)}"
            )
        for det in self.detections:
            if det.obb is None:
                raise ValueError(
                    f"Detection {det.object_id} in OBBDetectorOutput is missing required 'obb' field"
                )
        return self


# ============================================================================
# 4. Segmentation_Grounding Contracts
# ============================================================================

GroundingPipeline = Literal[
    "remote_sam", "remote_sam_only", "remote_sam_sam3_v1", "remote_sam_sam3_v2"
]


class GroundingResult(StrictModel):
    """Grounded referent entity representing segmented and localized object."""
    object_id: str = Field(description="Unique object instance identifier")
    class_name: Optional[str] = Field(default=None, description="Inferred object class")
    referring_target: Optional[str] = Field(default=None, description="Referring expression prompt")
    confidence: Confidence = Field(description="Grounding confidence")
    bbox_px: BBoxPx = Field(description="AABB bounding box for localization and benchmark evaluation")
    obb: Optional[OBB] = Field(default=None, description="Derived oriented bounding box")
    polygon: Optional[list[list[float]]] = Field(
        default=None, description="Contour polygon points [[x1, y1], [x2, y2], ...]"
    )
    mask_artifact: Optional[ArtifactRef] = Field(
        default=None, description="External artifact reference to full resolution binary mask"
    )
    source_model: Optional[str] = Field(default=None, description="Model pipeline identifier")
    candidate_info: Optional[dict[str, Any]] = Field(default=None, description="Candidate diagnostics")

    @property
    def mask_ref(self) -> Optional[ArtifactRef]:
        """Convenience property for mask_artifact."""
        return self.mask_artifact


class SegmentationGroundingInput(ToolInput):
    """Input contract for Segmentation_Grounding tool."""
    asset: AssetRef = Field(description="Target image asset reference")
    text_prompt: str = Field(description="Referring expression or target object description")
    roi: Optional[BBoxPx] = Field(default=None, description="Optional spatial constraint")
    pipeline: GroundingPipeline = Field(
        default="remote_sam_sam3_v2",
        description="Segmentation grounding pipeline variant",
    )
    return_polygon: bool = Field(default=True, description="Whether to extract contour polygons")
    return_obb: bool = Field(default=True, description="Whether to compute minimum area oriented box")
    confidence_threshold: float = Field(
        default=0.35, ge=0.0, le=1.0, description="Minimum mask confidence cutoff"
    )


class SegmentationGroundingOutput(ToolOutput):
    """Output contract from Segmentation_Grounding tool."""
    groundings: list[GroundingResult] = Field(
        default_factory=list, description="Resolved grounded entities"
    )
    model_metadata: dict[str, Any] = Field(default_factory=dict, description="Metadata")
    artifacts: list[ArtifactRef] = Field(
        default_factory=list, description="Derived mask artifact references"
    )
