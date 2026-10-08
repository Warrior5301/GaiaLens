"""Unit tests for Object_Detector Pydantic input/output schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from sac_shared_contracts.common import AssetRef, BBoxPx, Confidence, Detection, Modality
from sac_shared_contracts.perception import (
    NMSConfig,
    ObjectDetectorInput,
    ObjectDetectorOutput,
    TilingConfig,
)
from sac_shared_contracts.roi import ROI


@pytest.fixture
def sample_asset() -> AssetRef:
    return AssetRef(
        asset_id="asset_schema_test",
        uri="examples/sample_image.png",
        modality=Modality.OPTICAL,
        width_px=1024,
        height_px=1024,
    )


def test_valid_object_detector_input(sample_asset: AssetRef):
    inp = ObjectDetectorInput(
        asset=sample_asset,
        target_classes=["airplane", "ship"],
        confidence_threshold=0.4,
        nms_config=NMSConfig(iou_threshold=0.5, score_threshold=0.3),
        tiling_config=TilingConfig(enabled=True, tile_size_px=512, overlap_px=64),
    )
    assert inp.asset.asset_id == "asset_schema_test"
    assert inp.target_classes == ["airplane", "ship"]
    assert inp.confidence_threshold == 0.4
    assert inp.tiling_config.enabled is True


def test_input_with_roi_instance_and_bbox(sample_asset: AssetRef):
    # Both BBoxPx and ROI instance must be accepted
    bbox = BBoxPx(x_min=10.0, y_min=20.0, x_max=200.0, y_max=300.0)
    inp1 = ObjectDetectorInput(asset=sample_asset, roi=bbox)
    assert inp1.roi == bbox

    roi = ROI(roi_id="roi_01", source_asset_id=sample_asset.asset_id, bbox_px=bbox)
    inp2 = ObjectDetectorInput(asset=sample_asset, roi=roi)
    assert inp2.roi == roi


def test_invalid_confidence_threshold(sample_asset: AssetRef):
    with pytest.raises(ValidationError):
        ObjectDetectorInput(asset=sample_asset, confidence_threshold=1.5)
    with pytest.raises(ValidationError):
        ObjectDetectorInput(asset=sample_asset, confidence_threshold=-0.1)


def test_invalid_bbox_ordering(sample_asset: AssetRef):
    with pytest.raises(ValidationError):
        BBoxPx(x_min=500.0, y_min=100.0, x_max=200.0, y_max=300.0)


def test_object_detector_output_count_consistency():
    det = Detection(
        object_id="det_01",
        class_name="airplane",
        confidence=Confidence(score=0.9, source="test"),
        bbox_px=BBoxPx(x_min=0, y_min=0, x_max=10, y_max=10),
    )
    # Valid count
    out = ObjectDetectorOutput(detections=[det], count=1)
    assert out.count == 1

    # Count mismatch must raise ValidationError
    with pytest.raises(ValidationError, match="count mismatch"):
        ObjectDetectorOutput(detections=[det], count=5)
