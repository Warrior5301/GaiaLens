"""Object_Detector tool implementation for SAC GeoAI Agent.

Performs closed-set AABB object detection, supports object counting (count = len(detections)),
supports optional ROI-restricted detection with configurable policy,
supports tiled inference for large imagery, produces evidence and artifacts,
and integrates with Geo and ROI pipelines.
"""

from __future__ import annotations

import time
from typing import Any, Optional
import uuid
from app.core.artifacts import ArtifactStore
from app.core.logging import logger
from app.core.runtime import VisionRuntime
from app.tools.base import BaseTool, ToolExecutionException
from app.tools.object_detector.backend import (
    DetectorBackend,
    MockDetectorBackend,
    RemoteSensingYOLOBackend,
)
from app.tools.object_detector.class_mapping import ClassMapper
from app.tools.object_detector.postprocess import (
    ROIPolicy,
    apply_nms,
    assign_object_ids,
    confidence_filter,
    filter_roi,
)
from app.tools.object_detector.processor import ObjectDetectorProcessor
from app.tools.object_detector.tiling import TilingStrategy
from sac_shared_contracts.common import (
    ArtifactRef,
    BBoxPx,
    Confidence,
    Detection,
    PointPx,
)
from sac_shared_contracts.evidence import Evidence, EvidenceType
from sac_shared_contracts.geo_spatial import GeometryRef
from sac_shared_contracts.perception import ObjectDetectorInput, ObjectDetectorOutput
from sac_shared_contracts.tool_protocol import ToolError, ToolErrorCode


class ObjectDetector(BaseTool[ObjectDetectorInput, ObjectDetectorOutput]):
    """Production-grade closed-set Axis-Aligned Bounding Box Object Detector."""

    name = "Object_Detector"
    version = "1.0.0"
    description = "Deterministic closed-set detector for object localization, counting, and spatial enumeration"
    input_schema_class = ObjectDetectorInput
    output_schema_class = ObjectDetectorOutput

    def __init__(
        self,
        runtime: Optional[VisionRuntime] = None,
        model_backend: Optional[DetectorBackend] = None,
        artifact_store: Optional[ArtifactStore] = None,
        default_roi_policy: ROIPolicy = "center_inside",
    ) -> None:
        self.runtime = runtime or VisionRuntime()
        self.backend = model_backend or RemoteSensingYOLOBackend()
        self.artifact_store = artifact_store or ArtifactStore()
        self.default_roi_policy: ROIPolicy = default_roi_policy

    def execute(self, input_data: ObjectDetectorInput) -> ObjectDetectorOutput:
        """Run object detection on asset, applying validation, tiling, NMS, and ROI filtering."""
        start_time = time.perf_counter()
        request_id = f"req_det_{uuid.uuid4().hex[:8]}"

        # 1. Load image via VisionRuntime
        image, roi_box = ObjectDetectorProcessor.prepare_image(input_data, self.runtime)
        img_w, img_h = image.size

        # 2. Strict Semantic Validation (Dimensions, threshold, overlap, and ROI bounds)
        val_error = ObjectDetectorProcessor.validate_inputs(
            input_data=input_data,
            image_width=img_w,
            image_height=img_h,
        )
        if val_error is not None:
            logger.error(f"[{self.name}] Input validation error: {val_error.message}")
            raise ToolExecutionException(val_error)

        # 3. Class vocabulary normalization
        target_classes = ClassMapper.normalize_target_classes(input_data.target_classes)
        logger.info(
            f"[{self.name}] Starting detection (request_id='{request_id}', asset='{input_data.asset.asset_id}', "
            f"size={img_w}x{img_h}, target_classes={target_classes}, roi={roi_box is not None})"
        )

        # 4. Determine execution strategy (Tiled vs Full-frame)
        tiling = input_data.tiling_config
        use_tiling = tiling.enabled or (img_w > 1024 or img_h > 1024)
        tile_count = 1
        raw_candidates: list[Detection] = []

        if use_tiling:
            tile_size = tiling.tile_size_px if tiling.enabled else 512
            overlap_ratio = (
                float(tiling.overlap_px) / float(tile_size)
                if tiling.enabled and tile_size > 0
                else 0.15
            )
            tiling_strategy = TilingStrategy(
                tile_size_px=tile_size,
                overlap_ratio=overlap_ratio,
                iou_threshold=input_data.nms_config.iou_threshold,
            )

            # Local patch detector callable
            def _patch_detect(patch_img) -> list[Detection]:
                return self.backend.predict(
                    image=patch_img,
                    target_classes=target_classes,
                    confidence_threshold=input_data.confidence_threshold,
                )

            merged_dets, tile_metrics = tiling_strategy.run_tiled_inference(image, _patch_detect)
            tile_count = int(tile_metrics.get("tile_count", 1))
            raw_candidates = merged_dets
        else:
            raw_candidates = self.backend.predict(
                image=image,
                target_classes=target_classes,
                confidence_threshold=input_data.confidence_threshold,
            )

        count_before_nms = len(raw_candidates)

        # 5. Confidence filtering
        confident_dets = confidence_filter(raw_candidates, input_data.confidence_threshold)

        # 6. Non-Maximum Suppression (NMS)
        nms_survivors = apply_nms(
            detections=confident_dets,
            iou_threshold=input_data.nms_config.iou_threshold,
            class_agnostic=False,
        )

        # 7. ROI Spatial Filtering (if ROI is present)
        if roi_box is not None:
            post_roi_dets = filter_roi(
                detections=nms_survivors,
                roi=roi_box,
                policy=self.default_roi_policy,
                intersection_ratio_threshold=0.3,
            )
        else:
            post_roi_dets = nms_survivors

        # 8. Assign deterministic object IDs
        final_detections = assign_object_ids(post_roi_dets, prefix="det")

        # 9. Derive cardinality count (EXACT rule: count = len(valid_detections))
        final_count = len(final_detections)

        # 10. Persist detection set artifact
        artifact_ref = self.artifact_store.save_detection_set(
            detections_data=[d.model_dump() for d in final_detections],
            source_asset_id=input_data.asset.asset_id,
        )

        latency_ms = (time.perf_counter() - start_time) * 1000.0

        # 11. Model & Runtime metadata
        backend_meta = self.backend.get_metadata()
        model_metadata = {
            "model_id": backend_meta.get("model_id", self.backend.model_id),
            "model_version": getattr(self.backend, "version", "1.0.0"),
            "architecture": getattr(self.backend, "ARCH_NAME", "YOLO-VHR"),
            "checkpoint_path": backend_meta.get("checkpoint_path"),
            "device": backend_meta.get("device", self.backend.device),
            "precision": backend_meta.get("precision", self.backend.precision),
            "input_resolution": [img_w, img_h],
            "artifact_id": artifact_ref.artifact_id,
        }

        processing_metrics = {
            "request_id": request_id,
            "latency_ms": round(latency_ms, 2),
            "tiling_enabled": use_tiling,
            "tile_count": tile_count,
            "detection_count_before_nms": count_before_nms,
            "detection_count_after_nms": len(nms_survivors),
            "final_detection_count": final_count,
            "roi_applied": roi_box is not None,
            "roi_policy": self.default_roi_policy if roi_box else None,
        }

        # 12. Structured Logging
        logger.info(
            f"[{self.name}] Completed detection (request_id='{request_id}', asset='{input_data.asset.asset_id}', "
            f"found={final_count} detections, latency={latency_ms:.2f}ms, tiles={tile_count})"
        )

        return ObjectDetectorOutput(
            detections=final_detections,
            count=final_count,
            model_metadata=model_metadata,
            processing_metrics=processing_metrics,
        )

    def to_evidence(self, output: ObjectDetectorOutput, asset_id: str) -> list[Evidence]:
        """Convert detector execution results into structured, traceable Evidence."""
        evidences: list[Evidence] = []
        target_classes = sorted(list({d.class_name for d in output.detections}))

        # Primary detection enumeration evidence
        evidences.append(
            Evidence(
                evidence_id=f"ev_det_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.DETECTION,
                source_tool=self.name,
                source_tool_version=self.version,
                asset_id=asset_id,
                payload={
                    "target_classes": target_classes,
                    "count": output.count,
                    "model_id": output.model_metadata.get("model_id"),
                },
                confidence=0.92 if output.count > 0 else 0.85,
            )
        )

        # Secondary count evidence
        evidences.append(
            Evidence(
                evidence_id=f"ev_cnt_{uuid.uuid4().hex[:8]}",
                evidence_type=EvidenceType.COUNT,
                source_tool=self.name,
                source_tool_version=self.version,
                asset_id=asset_id,
                payload={"count": output.count, "classes": target_classes},
                confidence=0.95,
            )
        )

        return evidences

    @staticmethod
    def to_geometry_ref(detection: Detection) -> GeometryRef:
        """Convert a detection into a canonical GeometryRef for Team B (Geo/Spatial) interoperability."""
        return GeometryRef(
            geometry_type="bbox_px",
            bbox_px=detection.bbox_px,
        )
