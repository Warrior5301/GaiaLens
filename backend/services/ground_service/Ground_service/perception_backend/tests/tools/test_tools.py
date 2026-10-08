"""Tests for tool runtime execution, schema validation, and evidence generation."""

from __future__ import annotations

from app.models.registry import ToolRegistry
from app.schemas.common import AssetRef
from app.schemas.evidence import EvidenceType
from app.schemas.perception import (
    ImageAnalyzerInput,
    OBBDetectorInput,
    ObjectDetectorInput,
    SegmentationGroundingInput,
)


def get_test_asset() -> AssetRef:
    from pathlib import Path
    img = "test_assets/default_satellite.png"
    if not Path(img).exists():
        img = "examples/real_satellite/05873_0000.png"
    return AssetRef(
        asset_id="test_img",
        uri=img,
        modality="optical",
        width_px=512,
        height_px=512,
    )


def test_image_analyzer_execution():
    registry = ToolRegistry("config/tool_registry.yaml")
    tool = registry.instantiate_tool("Image_Analyzer")

    inp = ImageAnalyzerInput(asset=get_test_asset(), task_type="caption")
    res = tool.run(inp)

    assert res.status == "success"
    assert res.data is not None
    assert "caption" in res.data
    assert res.data["caption"] is not None

    # Test evidence generation
    validated_out = tool.output_schema_class.model_validate(res.data)
    evs = tool.to_evidence(validated_out, "test_img")
    assert len(evs) >= 1
    assert any(ev.evidence_type == EvidenceType.CAPTION for ev in evs)


def test_object_detector_execution():
    registry = ToolRegistry("config/tool_registry.yaml")
    tool = registry.instantiate_tool("Object_Detector")

    inp = ObjectDetectorInput(asset=get_test_asset(), target_classes=["airplane"])
    res = tool.run(inp)

    assert res.status == "success"
    assert res.data is not None
    assert res.data["count"] == len(res.data["detections"])

    # Test evidence generation
    validated_out = tool.output_schema_class.model_validate(res.data)
    evs = tool.to_evidence(validated_out, "test_img")
    assert any(ev.evidence_type == EvidenceType.COUNT for ev in evs)


def test_obb_detector_execution():
    registry = ToolRegistry("config/tool_registry.yaml")
    tool = registry.instantiate_tool("OBB_Detector")

    inp = OBBDetectorInput(asset=get_test_asset(), target_classes=["ship"])
    res = tool.run(inp)

    assert res.status == "success"
    assert res.data is not None
    assert res.data["count"] == len(res.data["detections"])
    for det in res.data["detections"]:
        assert det["obb"] is not None
        assert det["obb"]["angle_convention"] == "ccw_from_image_x"


def test_segmentation_grounding_execution():
    registry = ToolRegistry("config/tool_registry.yaml")
    tool = registry.instantiate_tool("Segmentation_Grounding")

    inp = SegmentationGroundingInput(
        asset=get_test_asset(),
        text_prompt="tennis court",
        pipeline="remote_sam_sam3_v2",
    )
    res = tool.run(inp)

    assert res.status == "success"
    assert res.data is not None
    assert len(res.data["groundings"]) > 0
    g = res.data["groundings"][0]
    assert g["bbox_px"] is not None
    assert g["mask_artifact"] is not None
