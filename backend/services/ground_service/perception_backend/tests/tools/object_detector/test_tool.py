"""Unit tests for ObjectDetector tool execution, evidence, and artifacts."""

from __future__ import annotations

import pytest
from app.core.runtime import VisionRuntime
from app.tools.object_detector.backend import MockDetectorBackend
from app.tools.object_detector.tool import ObjectDetector
from sac_shared_contracts.common import AssetRef, BBoxPx, Confidence, Detection, Modality
from sac_shared_contracts.perception import ObjectDetectorInput
from sac_shared_contracts.tool_protocol import ToolErrorCode, ToolExecutionStatus


@pytest.fixture
def test_asset() -> AssetRef:
    return AssetRef(
        asset_id="tool_test_asset",
        uri="examples/sample_image.png",
        modality=Modality.OPTICAL,
        width_px=512,
        height_px=512,
    )


def test_object_detector_execution_with_mock_backend(test_asset: AssetRef):
    runtime = VisionRuntime()
    mock_backend = MockDetectorBackend()
    detector = ObjectDetector(runtime=runtime, model_backend=mock_backend)

    inp = ObjectDetectorInput(
        asset=test_asset,
        target_classes=["airplane"],
        confidence_threshold=0.35,
    )

    res = detector.run(inp)
    assert res.status == ToolExecutionStatus.SUCCESS
    assert res.data is not None

    detections = res.data["detections"]
    count = res.data["count"]
    assert count == len(detections)
    assert count > 0

    # Verify model metadata and metrics
    meta = res.data["model_metadata"]
    assert "model_id" in meta
    assert "artifact_id" in meta

    metrics = res.data["processing_metrics"]
    assert "latency_ms" in metrics
    assert metrics["final_detection_count"] == count


def test_object_detector_invalid_roi_exceeds_image(test_asset: AssetRef):
    runtime = VisionRuntime()
    detector = ObjectDetector(runtime=runtime)

    # ROI exceeding image width 512
    bad_roi = BBoxPx(x_min=100.0, y_min=100.0, x_max=999.0, y_max=300.0)
    inp = ObjectDetectorInput(
        asset=test_asset,
        roi=bad_roi,
    )

    res = detector.run(inp)
    assert res.status == ToolExecutionStatus.FAILED
    assert len(res.errors) == 1
    assert res.errors[0].code == ToolErrorCode.INVALID_ROI


def test_object_detector_evidence_generation(test_asset: AssetRef):
    detector = ObjectDetector()
    inp = ObjectDetectorInput(asset=test_asset, target_classes=["airplane"])
    res = detector.run(inp)
    assert res.status == ToolExecutionStatus.SUCCESS

    # Convert to Evidence
    output = detector.validate_output(res.data)
    evidences = detector.to_evidence(output, asset_id=test_asset.asset_id)
    assert len(evidences) >= 2

    types = [ev.evidence_type for ev in evidences]
    assert "detection" in types
    assert "count" in types
    assert all(ev.source_tool == "Object_Detector" for ev in evidences)
