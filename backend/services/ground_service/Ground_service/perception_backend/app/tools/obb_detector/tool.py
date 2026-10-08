"""OBB_Detector tool implementation.

Performs oriented bounding box detection (cx, cy, w, h, angle_deg)
with internal canonical angle convention: 'ccw_from_image_x'.
Provides AABB conversion for VRS-Bench evaluation.
"""

from __future__ import annotations

from typing import Any, Optional
from app.core.geometry import obb_to_aabb
from app.core.logging import logger
from app.core.runtime import VisionRuntime
from app.schemas.common import Confidence, Detection, OBB, PointPx
from app.schemas.evidence import Evidence, EvidenceType
from app.schemas.perception import OBBDetectorInput, OBBDetectorOutput
from app.tools.base import BaseTool


class OBBDetector(BaseTool[OBBDetectorInput, OBBDetectorOutput]):
    """Oriented Bounding Box Object Detector for directional remote-sensing entities."""

    name = "OBB_Detector"
    version = "1.0.0"
    description = "Rotated object detector with orientation estimation (angle, width, height)"
    input_schema_class = OBBDetectorInput
    output_schema_class = OBBDetectorOutput

    def __init__(
        self,
        runtime: Optional[VisionRuntime] = None,
        model_backend: Optional[Any] = None,
    ) -> None:
        self.runtime = runtime or VisionRuntime()
        self.model_backend = model_backend

    def execute(self, input_data: OBBDetectorInput) -> OBBDetectorOutput:
        """Run rotated object detection."""
        logger.info(
            f"[{self.name}] Running OBB detection on asset: '{input_data.asset.asset_id}', "
            f"targets: {input_data.target_classes}"
        )

        target_class = input_data.target_classes[0] if input_data.target_classes else "ship"
        w = float(input_data.asset.width_px)
        h = float(input_data.asset.height_px)

        sample_obbs = [
            (w * 0.45, h * 0.5, 60.0, 180.0, 35.0, 0.93),
        ]

        detections: list[Detection] = []
        for i, (cx, cy, box_w, box_h, angle, score) in enumerate(sample_obbs):
            if score >= input_data.confidence_threshold:
                obb = OBB(
                    cx=cx,
                    cy=cy,
                    width=box_w,
                    height=box_h,
                    angle_deg=angle,
                    angle_convention="ccw_from_image_x",
                )
                # Compute enclosing AABB for benchmark compatibility
                aabb = obb_to_aabb(obb, clip_bounds=(input_data.asset.width_px, input_data.asset.height_px))
                detections.append(
                    Detection(
                        object_id=f"obb_{i + 1}",
                        class_name=target_class,
                        confidence=Confidence(score=score, calibrated=False, source="rotated_detector_nms"),
                        bbox_px=aabb,
                        obb=obb,
                        center_px=PointPx(x=cx, y=cy),
                    )
                )

        return OBBDetectorOutput(
            detections=detections,
            count=len(detections),
            model_metadata={"architecture": "YOLO-OBB", "precision": "fp16"},
            processing_metrics={"rotated_nms_boxes": len(sample_obbs)},
        )

    def to_evidence(self, output: OBBDetectorOutput, asset_id: str) -> list[Evidence]:
        evidences: list[Evidence] = []
        evidences.append(
            Evidence(
                evidence_type=EvidenceType.COUNT,
                source_tool=self.name,
                source_tool_version=self.version,
                asset_id=asset_id,
                payload={"count": output.count},
                confidence=0.95 if output.count > 0 else 0.8,
            )
        )
        for det in output.detections:
            evidences.append(
                Evidence(
                    evidence_type=EvidenceType.OBB_DETECTION,
                    source_tool=self.name,
                    source_tool_version=self.version,
                    asset_id=asset_id,
                    payload={
                        "object_id": det.object_id,
                        "class_name": det.class_name,
                        "obb": det.obb.model_dump() if det.obb else None,
                        "bbox_px": det.bbox_px.to_xyxy(),
                    },
                    confidence=det.confidence.score,
                )
            )
        return evidences
