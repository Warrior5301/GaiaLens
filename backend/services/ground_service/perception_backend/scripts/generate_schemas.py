"""Generates JSON Schemas (2020-12) from Pydantic models.

Ensures:
- Pydantic models are the single source of truth for runtime validation
- Generated JSON Schemas are machine-readable contracts stored in schemas/
- Validates each generated schema with jsonschema draft 2020-12 validator
"""

from __future__ import annotations

import json
from pathlib import Path
import sys

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from jsonschema import Draft202012Validator
from app.schemas.agent import AgentPlan, Claim, FinalResponse, PlanStep
from app.schemas.common import (
    ArtifactRef,
    AssetRef,
    BBoxPx,
    Confidence,
    Detection,
    MaskArtifact,
    OBB,
    PointPx,
    ROI,
)
from app.schemas.evidence import Evidence
from app.schemas.perception import (
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
from app.schemas.tool_protocol import ToolCall, ToolError, ToolExecutionResult

SCHEMA_REGISTRY: dict[str, type] = {
    # Common & Domain schemas
    "common": AssetRef,
    "point_px": PointPx,
    "bbox_px": BBoxPx,
    "obb": OBB,
    "confidence": Confidence,
    "asset_ref": AssetRef,
    "artifact_ref": ArtifactRef,
    "mask_artifact": MaskArtifact,
    "roi": ROI,
    "detection": Detection,
    # Tool protocol schemas
    "tool_call": ToolCall,
    "tool_error": ToolError,
    "tool_execution_result": ToolExecutionResult,
    # Perception tool contracts
    "image_analyzer.input": ImageAnalyzerInput,
    "image_analyzer.output": ImageAnalyzerOutput,
    "object_detector.input": ObjectDetectorInput,
    "object_detector.output": ObjectDetectorOutput,
    "obb_detector.input": OBBDetectorInput,
    "obb_detector.output": OBBDetectorOutput,
    "grounding_result": GroundingResult,
    "segmentation_grounding.input": SegmentationGroundingInput,
    "segmentation_grounding.output": SegmentationGroundingOutput,
    # Evidence & Agent contracts
    "evidence": Evidence,
    "plan_step": PlanStep,
    "agent_plan": AgentPlan,
    "claim": Claim,
    "final_response": FinalResponse,
}


def generate_all_schemas(output_dir: str | Path = "schemas") -> int:
    """Generate and validate all schema files."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    print(f"Generating {len(SCHEMA_REGISTRY)} JSON Schemas into '{out_path.resolve()}'...")
    errors: list[str] = []

    for name, model_cls in SCHEMA_REGISTRY.items():
        schema_file = out_path / f"{name}.schema.json"
        schema_dict = model_cls.model_json_schema()

        # Add Draft 2020-12 identifier and schema metadata
        schema_dict["$schema"] = "https://json-schema.org/draft/2020-12/schema"
        schema_dict["$id"] = f"https://sac.geoai.org/schemas/{name}.schema.json"
        schema_dict["title"] = model_cls.__name__

        # Validate schema self-consistency against Draft202012 meta-schema
        try:
            Draft202012Validator.check_schema(schema_dict)
        except Exception as e:
            err_msg = f"Meta-schema validation failed for {name}: {e}"
            print(f"ERROR: {err_msg}", file=sys.stderr)
            errors.append(err_msg)
            continue

        with open(schema_file, "w", encoding="utf-8") as f:
            json.dump(schema_dict, f, indent=2)

        print(f"  [OK] {schema_file.name} (from {model_cls.__name__})")

    if errors:
        print(f"\nCompleted with {len(errors)} errors.", file=sys.stderr)
        return 1

    print(f"\nSuccessfully generated and verified all {len(SCHEMA_REGISTRY)} schemas.")
    return 0


if __name__ == "__main__":
    sys.exit(generate_all_schemas())
