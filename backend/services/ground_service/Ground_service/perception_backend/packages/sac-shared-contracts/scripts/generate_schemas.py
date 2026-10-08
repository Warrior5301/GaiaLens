"""Automated JSON Schema generator for SAC Shared Contracts.

Generates Draft 2020-12 compliant JSON Schemas from Pydantic models.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure package src is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from jsonschema import Draft202012Validator

from sac_shared_contracts.agent import AgentPlan, Claim, FinalResponse, PlanStep
from sac_shared_contracts.common import (
    ArtifactRef,
    AssetRef,
    BBoxPx,
    Confidence,
    Detection,
    EvidenceRef,
    GeoPoint,
    OBB,
    PointPx,
)
from sac_shared_contracts.evidence import Evidence
from sac_shared_contracts.geo_spatial import (
    GeoMeasurementInput,
    GeoMeasurementOutput,
    GeometryRef,
    MeasurementValue,
    SpatialRelationship,
    SpatialRelationshipInput,
    SpatialRelationshipOutput,
)
from sac_shared_contracts.multimodal import (
    AlignmentReport,
    FusionInput,
    FusionOutput,
    ModalityAsset,
)
from sac_shared_contracts.ocr import OCRInput, OCROutput, OCRTextRegion
from sac_shared_contracts.perception import (
    GroundingResult,
    ImageAnalyzerInput,
    ImageAnalyzerOutput,
    OBBDetectorInput,
    OBBDetectorOutput,
    ObjectDetectorInput,
    ObjectDetectorOutput,
    SegmentationGroundingInput,
    SegmentationGroundingOutput,
)
from sac_shared_contracts.roi import ROI, ROILocalizerInput, ROILocalizerOutput
from sac_shared_contracts.tool_protocol import (
    ToolCall,
    ToolError,
    ToolExecutionResult,
)

SCHEMA_VERSION = "0.1.1"

CONTRACT_MODELS: dict[str, type] = {
    # Common
    "asset_ref": AssetRef,
    "point_px": PointPx,
    "bbox_px": BBoxPx,
    "geo_point": GeoPoint,
    "obb": OBB,
    "confidence": Confidence,
    "artifact_ref": ArtifactRef,
    "evidence_ref": EvidenceRef,
    "detection": Detection,
    # Protocol
    "tool_call": ToolCall,
    "tool_execution_result": ToolExecutionResult,
    "tool_error": ToolError,
    # Perception (Team A)
    "image_analyzer.input": ImageAnalyzerInput,
    "image_analyzer.output": ImageAnalyzerOutput,
    "object_detector.input": ObjectDetectorInput,
    "object_detector.output": ObjectDetectorOutput,
    "obb_detector.input": OBBDetectorInput,
    "obb_detector.output": OBBDetectorOutput,
    "segmentation_grounding.input": SegmentationGroundingInput,
    "segmentation_grounding.output": SegmentationGroundingOutput,
    "grounding_result": GroundingResult,
    # Geo/Spatial (Team B)
    "geometry_ref": GeometryRef,
    "measurement_value": MeasurementValue,
    "geo_measurement.input": GeoMeasurementInput,
    "geo_measurement.output": GeoMeasurementOutput,
    "spatial_relationship": SpatialRelationship,
    "spatial_relationship.input": SpatialRelationshipInput,
    "spatial_relationship.output": SpatialRelationshipOutput,
    # ROI & Multimodal & OCR (Team C)
    "roi": ROI,
    "roi_localizer.input": ROILocalizerInput,
    "roi_localizer.output": ROILocalizerOutput,
    "modality_asset": ModalityAsset,
    "alignment_report": AlignmentReport,
    "fusion.input": FusionInput,
    "fusion.output": FusionOutput,
    "ocr_text_region": OCRTextRegion,
    "ocr.input": OCRInput,
    "ocr.output": OCROutput,
    # Evidence & Agent
    "evidence": Evidence,
    "plan_step": PlanStep,
    "agent_plan": AgentPlan,
    "claim": Claim,
    "final_response": FinalResponse,
}


def generate_all_schemas(output_dir: str | Path | None = None) -> int:
    out_path = Path(output_dir) if output_dir else REPO_ROOT / "schemas" / "generated"
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(CONTRACT_MODELS)} JSON Schemas into '{out_path.resolve()}'...")
    errors: list[str] = []
    seen_ids: set[str] = set()

    for name, model_cls in CONTRACT_MODELS.items():
        schema_file = out_path / f"{name}.schema.json"
        schema_id = f"https://sac.geoai.org/schemas/v{SCHEMA_VERSION}/{name}.schema.json"

        if schema_id in seen_ids:
            errors.append(f"Duplicate schema $id detected: {schema_id}")
        seen_ids.add(schema_id)

        schema_dict = model_cls.model_json_schema()
        schema_dict["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema_dict["$id"] = schema_id
        schema_dict["title"] = model_cls.__name__
        schema_dict["x-schema-version"] = SCHEMA_VERSION

        # Verify against Draft 2020-12 meta-schema
        try:
            Draft202012Validator.check_schema(schema_dict)
        except Exception as e:
            err = f"Meta-schema check failed for '{name}': {e}"
            print(f"  [ERROR] {err}", file=sys.stderr)
            errors.append(err)
            continue

        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)

        print(f"  [OK] {schema_file.name} (from {model_cls.__name__})")

    if errors:
        print(f"\nSchema generation failed with {len(errors)} error(s).", file=sys.stderr)
        return 1

    print(f"\nAll {len(CONTRACT_MODELS)} JSON Schemas successfully generated and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(generate_all_schemas())
