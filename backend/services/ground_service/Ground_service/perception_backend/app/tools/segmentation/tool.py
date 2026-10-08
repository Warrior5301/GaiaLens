"""Segmentation_Grounding tool implementation for SAC GeoAI Agent.

Performs text-guided visual grounding and pixel-level segmentation
following the RemoteSAM-SAM3-v2 research architecture.
"""

from __future__ import annotations

import logging
from typing import Any, Optional
import numpy as np
from PIL import Image

from app.core.artifacts import ArtifactStore
from app.core.runtime import VisionRuntime
from app.schemas.evidence import Evidence, EvidenceType
from app.schemas.perception import (
    SegmentationGroundingInput,
    SegmentationGroundingOutput,
)
from app.schemas.tool_protocol import ToolError, ToolErrorCode
from app.tools.base import BaseTool, ToolExecutionException
from app.tools.segmentation.backend import (
    MockRemoteSAMBackend,
    MockSAM3Backend,
    RemoteSAMBackend,
    SAM3Backend,
)
from app.tools.segmentation.backends.segearth_ov3 import SegEarthOV3Backend
from app.tools.segmentation.prompt_parser import parse_multi_target_prompt
from app.tools.segmentation.pipeline import GroundingPipelineOrchestrator

logger = logging.getLogger("app.tools.segmentation.tool")



class SegmentationGrounding(BaseTool[SegmentationGroundingInput, SegmentationGroundingOutput]):
    """Production-style text-guided visual grounding and segmentation tool."""

    name = "Segmentation_Grounding"
    version = "1.0.0"
    description = (
        "Grounds natural language referring expressions to pixel-level segmentation masks, "
        "AABBs, and OBBs via RemoteSAM and SAM3 refinement pipelines."
    )
    input_schema_class = SegmentationGroundingInput
    output_schema_class = SegmentationGroundingOutput

    def __init__(
        self,
        runtime: Optional[VisionRuntime] = None,
        artifact_store: Optional[ArtifactStore] = None,
        model_backend: Optional[Any] = None,
        remote_sam_backend: Optional[RemoteSAMBackend] = None,
        sam3_backend: Optional[SAM3Backend] = None,
        dbscan_eps: float = 40.0,
        dbscan_min_samples: int = 1,
        logit_threshold: float = 0.5,
        min_component_area: int = 10,
    ) -> None:
        self.runtime = runtime or VisionRuntime()
        self.artifact_store = artifact_store or ArtifactStore()

        # If a single generic model_backend was passed (e.g. from registry or test)
        if model_backend is not None:
            if isinstance(model_backend, RemoteSAMBackend):
                remote_sam_backend = model_backend
            elif isinstance(model_backend, SAM3Backend):
                sam3_backend = model_backend

        # Fallback to mock backends if running in environments without GPU/weights
        r_backend = remote_sam_backend or MockRemoteSAMBackend()
        s_backend = sam3_backend or MockSAM3Backend()

        self.orchestrator = GroundingPipelineOrchestrator(
            remote_sam_backend=r_backend,
            sam3_backend=s_backend,
            artifact_store=self.artifact_store,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            logit_threshold=logit_threshold,
            min_component_area=min_component_area,
        )

    def execute(self, input_data: SegmentationGroundingInput) -> SegmentationGroundingOutput:
        """Run text-guided segmentation and visual grounding on the target asset."""
        logger.info(
            f"[{self.name}] Running visual grounding for '{input_data.text_prompt}' "
            f"using pipeline '{input_data.pipeline}' on asset '{input_data.asset.asset_id}'"
        )

        # 1. Load image via VisionRuntime
        try:
            image = self.runtime.load_image(input_data.asset)
        except Exception as exc:
            logger.error(f"[{self.name}] Failed to load image asset: {exc}")
            raise ToolExecutionException(
                ToolError(
                    code=ToolErrorCode.ASSET_LOAD_FAILED,
                    message=f"Failed to load image asset {input_data.asset.asset_id}: {exc}",
                )
            ) from exc

        img_w, img_h = image.size

        # 2. Semantic validation of optional ROI
        if input_data.roi is not None:
            roi = input_data.roi
            if roi.x_min >= roi.x_max or roi.y_min >= roi.y_max:
                raise ToolExecutionException(
                    ToolError(
                        code=ToolErrorCode.INVALID_INPUT,
                        message=f"Degenerate ROI bbox: x_min={roi.x_min}, x_max={roi.x_max}, y_min={roi.y_min}, y_max={roi.y_max}",
                    )
                )
            if roi.x_max < 0 or roi.y_max < 0 or roi.x_min > img_w or roi.y_min > img_h:
                raise ToolExecutionException(
                    ToolError(
                        code=ToolErrorCode.INVALID_INPUT,
                        message=f"ROI is entirely outside image bounds: image size ({img_w}, {img_h}) vs ROI {roi}",
                    )
                )

        # 3. Execute real SegEarth-OV3 open-vocabulary perception backend
        try:
            if not hasattr(self, "segearth_backend"):
                self.segearth_backend = SegEarthOV3Backend(confidence_threshold=input_data.confidence_threshold)

            targets = parse_multi_target_prompt(input_data.text_prompt)
            logger.info(f"[{self.name}] Multi-target decomposed prompts: {targets}")

            masks, boxes, scores, labels = self.segearth_backend.segment(
                image=image,
                text_prompts=targets,
                confidence_threshold=input_data.confidence_threshold,
            )

            groundings = self.segearth_backend.to_grounding_results(
                masks=masks,
                boxes=boxes,
                scores=scores,
                labels=labels,
                return_polygon=input_data.return_polygon,
                return_obb=input_data.return_obb,
            )

            artifacts = []
            if self.artifact_store and masks:
                for idx, (m, g) in enumerate(zip(masks, groundings)):
                    try:
                        mask_ref = self.artifact_store.save_mask(
                            mask=m,
                            source_asset_id=input_data.asset.asset_id,
                            metadata={"object_id": g.object_id, "class_name": g.class_name},
                        )
                        g.mask_artifact = mask_ref
                        artifacts.append(mask_ref)
                    except Exception as e:
                        logger.warning(f"Could not persist mask artifact: {e}")

            return SegmentationGroundingOutput(
                groundings=groundings,
                model_metadata=self.segearth_backend.get_metadata(),
                artifacts=artifacts,
            )
        except Exception as real_err:
            logger.error(f"[{self.name}] REAL MODEL GUARANTEE: SegEarthOV3 failed: {real_err}", exc_info=True)
            raise ToolExecutionException(
                ToolError(
                    code=ToolErrorCode.INTERNAL_EXECUTION_ERROR,
                    message=f"SegEarthOV3 execution failed: {real_err}. Mock backends are prohibited.",
                )
            ) from real_err



    def to_evidence(self, output: SegmentationGroundingOutput, asset_id: str) -> list[Evidence]:
        """Convert grounding results into verifiable Evidence records."""
        evidences: list[Evidence] = []
        for g in output.groundings:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.GROUNDING,
                    source_tool=self.name,
                    source_tool_version=self.version,
                    asset_id=asset_id,
                    payload={
                        "object_id": g.object_id,
                        "class_name": g.class_name,
                        "referring_target": g.referring_target,
                        "bbox_px": g.bbox_px.to_xyxy(),
                        "obb": g.obb.model_dump() if g.obb else None,
                        "mask_artifact_id": g.mask_artifact.artifact_id if g.mask_artifact else None,
                        "source_model": g.source_model,
                    },
                    confidence=g.confidence.score,
                )
            )
        return evidences
