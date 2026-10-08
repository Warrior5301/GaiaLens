"""Comprehensive contract verification script for SAC Shared Contracts.

Verifies:
1. Import of all Pydantic models.
2. Correct instantiation of valid payloads.
3. Strict rejection of invalid/malformed payloads.
4. Discrimination validation (e.g. GeometryRef).
5. Validation of all committed JSON examples in examples/.
6. Schema generation check without duplicate IDs.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
from pydantic import ValidationError

# Ensure package src and scripts are on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import sac_shared_contracts as ssc
from scripts.generate_schemas import generate_all_schemas


def validate_all() -> int:
    print("\n================ Validating SAC Shared Contracts ================")
    errors: list[str] = []

    # 1. Verify Imports
    print("1. Checking package exports...")
    expected_models = [
        "AssetRef", "ArtifactRef", "BBoxPx", "OBB", "PointPx", "GeoPoint", "Confidence",
        "Detection", "ToolCall", "ToolExecutionResult", "ToolError", "GeometryRef",
        "MeasurementValue", "GeoMeasurementInput", "SpatialRelationshipInput",
        "ROI", "ROILocalizerInput", "ModalityAsset", "FusionInput", "OCRTextRegion",
        "OCRInput", "Evidence", "AgentPlan", "FinalResponse"
    ]
    for model_name in expected_models:
        if not hasattr(ssc, model_name):
            errors.append(f"Missing exported model: {model_name}")
    print(f"   [OK] Verified {len(expected_models)} core models exported.")

    # 2. Test Valid Models
    print("2. Testing instantiation of valid domain entities...")
    try:
        asset = ssc.AssetRef(
            asset_id="asset_001",
            uri="s3://imagery/sample.tif",
            modality=ssc.Modality.OPTICAL,
            width_px=1024,
            height_px=1024,
            gsd_m=0.3,
            crs="EPSG:3857",
        )
        bbox = ssc.BBoxPx(x_min=10.0, y_min=20.0, x_max=100.0, y_max=200.0)
        obb = ssc.OBB(cx=55.0, cy=110.0, width=50.0, height=120.0, angle_deg=30.0)
        conf = ssc.Confidence(score=0.95, calibrated=True, source="benchmark_model")
        det = ssc.Detection(object_id="det_1", class_name="airplane", confidence=conf, bbox_px=bbox, obb=obb)

        # GeometryRef discriminated forms
        geom_bbox = ssc.GeometryRef(geometry_type="bbox_px", bbox_px=bbox)
        geom_poly = ssc.GeometryRef(geometry_type="polygon", polygon=[[0, 0], [10, 0], [10, 10], [0, 0]])
        geom_pt = ssc.GeometryRef(geometry_type="point", point=ssc.PointPx(x=5, y=5))
        geom_art = ssc.GeometryRef(geometry_type="artifact", artifact_id="mask_123")

        roi = ssc.ROI(roi_id="roi_01", source_asset_id=asset.asset_id, bbox_px=bbox)
        print("   [OK] Valid entities instantiated successfully.")
    except Exception as e:
        errors.append(f"Valid instantiation failed: {e}")

    # 3. Test Invalid Rejections
    print("3. Testing strict validation failures for malformed payloads...")
    # A. Inverted BBox
    try:
        ssc.BBoxPx(x_min=100.0, y_min=20.0, x_max=50.0, y_max=200.0)
        errors.append("BBoxPx failed to reject x_max < x_min")
    except (ValueError, ValidationError):
        pass

    # B. Out of range confidence
    try:
        ssc.Confidence(score=1.5, source="test")
        errors.append("Confidence failed to reject score > 1.0")
    except (ValueError, ValidationError):
        pass

    # C. GeometryRef missing discriminator field
    try:
        ssc.GeometryRef(geometry_type="bbox_px", bbox_px=None)
        errors.append("GeometryRef failed to require bbox_px when geometry_type='bbox_px'")
    except (ValueError, ValidationError):
        pass

    try:
        ssc.GeometryRef(geometry_type="artifact", artifact_id=None)
        errors.append("GeometryRef failed to require artifact_id when geometry_type='artifact'")
    except (ValueError, ValidationError):
        pass

    # D. StrictModel extra fields
    try:
        ssc.PointPx(x=10, y=20, extra_forbidden="bad")
        errors.append("StrictModel failed to forbid extra field")
    except ValidationError:
        pass

    print("   [OK] Malformed inputs strictly rejected.")

    # 4. Validate JSON Examples
    print("4. Validating files in examples/...")
    examples_dir = REPO_ROOT / "examples"
    example_mappings = [
        ("tool_call.json", ssc.ToolCall),
        ("tool_execution_result.json", ssc.ToolExecutionResult),
        ("roi_to_detector.json", ssc.ObjectDetectorInput),
        ("measurement_from_mask.json", ssc.GeoMeasurementInput),
        ("fusion_input.json", ssc.FusionInput),
        ("ocr_result.json", ssc.OCROutput),
    ]
    for filename, model_class in example_mappings:
        ex_file = examples_dir / filename
        if not ex_file.exists():
            errors.append(f"Missing example file: {ex_file}")
            continue
        try:
            with open(ex_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_class.model_validate(data)
            print(f"   [OK] {filename} successfully validates against {model_class.__name__}")
        except Exception as e:
            errors.append(f"Example {filename} failed validation against {model_class.__name__}: {e}")

    # 5. Schema Generation Check
    print("5. Verifying schema generation...")
    ret = generate_all_schemas()
    if ret != 0:
        errors.append("Schema generation script failed")

    print("=================================================================")
    if errors:
        print(f"\nVALIDATION FAILED with {len(errors)} error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    print("\nALL CONTRACTS, MODELS, SCHEMAS, AND EXAMPLES VALIDATED PERFECTLY.")
    return 0


if __name__ == "__main__":
    sys.exit(validate_all())
