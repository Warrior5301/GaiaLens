"""Integration tests for Segmentation_Grounding tool, ROI integration, and evidence."""

from __future__ import annotations

import pytest
from sac_shared_contracts.common import AssetRef, BBoxPx, Modality
from sac_shared_contracts.perception import (
    ObjectDetectorInput,
    SegmentationGroundingInput,
)
from sac_shared_contracts.roi import ROI
from sac_shared_contracts.tool_protocol import ToolErrorCode, ToolExecutionStatus

from app.core.artifacts import ArtifactStore
from app.core.runtime import VisionRuntime
from app.models.registry import ToolRegistry
from app.schemas.evidence import EvidenceType
from app.tools.object_detector.tool import ObjectDetector
from app.tools.segmentation.backend import MockRemoteSAMBackend, MockSAM3Backend
from app.tools.segmentation.tool import SegmentationGrounding


@pytest.fixture
def sample_asset() -> AssetRef:
    from pathlib import Path
    real_img = "examples/real_satellite/05873_0000.png"
    uri = real_img if Path(real_img).exists() else "examples/sample_image.png"
    return AssetRef(
        asset_id="grounding_test_asset",
        uri=uri,
        modality=Modality.OPTICAL,
        width_px=512,
        height_px=512,
    )


def test_segmentation_grounding_tool_execution(sample_asset: AssetRef, tmp_path):
    store = ArtifactStore(root_dir=str(tmp_path / "artifacts"))
    tool = SegmentationGrounding(artifact_store=store)

    inp = SegmentationGroundingInput(
        asset=sample_asset,
        text_prompt="tennis court",
        pipeline="remote_sam_sam3_v2",
        return_obb=True,
        return_polygon=True,
    )

    result = tool.run(inp)
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None

    data = result.data
    assert len(data["groundings"]) >= 1
    g = data["groundings"][0]
    assert g["bbox_px"] is not None
    assert g["obb"] is not None
    assert g["polygon"] is not None
    assert g["mask_artifact"] is not None

    # Validate output schema
    validated = tool.output_schema_class.model_validate(data)
    assert len(validated.groundings) >= 1

    # Verify Evidence Generation
    evidences = tool.to_evidence(validated, sample_asset.asset_id)
    assert len(evidences) >= 1
    assert evidences[0].evidence_type == EvidenceType.GROUNDING
    assert evidences[0].payload["class_name"] == "tennis court"


def test_roi_integration(sample_asset: AssetRef):
    """Test ROI_Localizer integration into SegmentationGroundingInput.roi."""
    roi_bbox = BBoxPx(x_min=50.0, y_min=100.0, x_max=300.0, y_max=400.0)
    simulated_roi = ROI(
        roi_id="roi_001",
        source_asset_id=sample_asset.asset_id,
        bbox_px=roi_bbox,
        reason="Potential aircraft apron",
    )

    tool = SegmentationGrounding()
    inp = SegmentationGroundingInput(
        asset=sample_asset,
        text_prompt="the airplane",
        roi=simulated_roi.bbox_px,
    )

    result = tool.run(inp)
    assert result.status == ToolExecutionStatus.SUCCESS
    assert result.data is not None
    assert result.data["model_metadata"]["pipeline"] == "remote_sam_sam3_v2"


def test_invalid_roi_rejected(sample_asset: AssetRef):
    tool = SegmentationGrounding()
    # ROI entirely outside image bounds (image is 512x512)
    inp = SegmentationGroundingInput(
        asset=sample_asset,
        text_prompt="the airplane",
        roi=BBoxPx(x_min=600.0, y_min=600.0, x_max=700.0, y_max=700.0),
    )

    result = tool.run(inp)
    assert result.status == ToolExecutionStatus.FAILED
    assert len(result.errors) > 0
    assert result.errors[0].code == ToolErrorCode.INVALID_INPUT


def test_object_detector_to_segmentation_integration(sample_asset: AssetRef):
    """Test using Object_Detector detection bbox as an ROI constraint for Segmentation_Grounding."""
    detector = ObjectDetector()
    det_inp = ObjectDetectorInput(
        asset=sample_asset,
        target_classes=["tennis-court"],
    )
    det_res = detector.run(det_inp)
    assert det_res.status == ToolExecutionStatus.SUCCESS
    assert len(det_res.data["detections"]) > 0

    first_det = det_res.data["detections"][0]
    det_box = BBoxPx.model_validate(first_det["bbox_px"])

    # Feed detection box as ROI to Segmentation_Grounding
    seg_tool = SegmentationGrounding()
    seg_inp = SegmentationGroundingInput(
        asset=sample_asset,
        text_prompt=f"the {first_det['class_name']}",
        roi=det_box,
    )
    seg_res = seg_tool.run(seg_inp)
    assert seg_res.status == ToolExecutionStatus.SUCCESS
    assert seg_res.data is not None
    assert len(seg_res.data["groundings"]) > 0
