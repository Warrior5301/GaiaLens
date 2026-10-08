"""Unit tests for pure postprocessing functions in Object_Detector."""

from __future__ import annotations

import pytest
from app.tools.object_detector.postprocess import (
    apply_nms,
    assign_object_ids,
    compute_iou,
    confidence_filter,
    filter_roi,
    restore_global_coordinates,
)
from sac_shared_contracts.common import BBoxPx, Confidence, Detection


@pytest.fixture
def sample_detections() -> list[Detection]:
    return [
        Detection(
            object_id="raw_1",
            class_name="airplane",
            confidence=Confidence(score=0.95, source="test"),
            bbox_px=BBoxPx(x_min=10.0, y_min=10.0, x_max=50.0, y_max=50.0),
        ),
        # Overlapping with raw_1 (IoU ~ 0.70)
        Detection(
            object_id="raw_2",
            class_name="airplane",
            confidence=Confidence(score=0.85, source="test"),
            bbox_px=BBoxPx(x_min=15.0, y_min=15.0, x_max=50.0, y_max=50.0),
        ),
        # Non-overlapping detection
        Detection(
            object_id="raw_3",
            class_name="airplane",
            confidence=Confidence(score=0.90, source="test"),
            bbox_px=BBoxPx(x_min=200.0, y_min=200.0, x_max=250.0, y_max=250.0),
        ),
        # Low confidence detection
        Detection(
            object_id="raw_4",
            class_name="airplane",
            confidence=Confidence(score=0.20, source="test"),
            bbox_px=BBoxPx(x_min=300.0, y_min=300.0, x_max=350.0, y_max=350.0),
        ),
    ]


def test_compute_iou():
    box1 = BBoxPx(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
    box2 = BBoxPx(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0)
    assert compute_iou(box1, box2) == 1.0

    box3 = BBoxPx(x_min=20.0, y_min=20.0, x_max=30.0, y_max=30.0)
    assert compute_iou(box1, box3) == 0.0


def test_confidence_filter(sample_detections: list[Detection]):
    filtered = confidence_filter(sample_detections, threshold=0.5)
    assert len(filtered) == 3
    assert all(d.confidence.score >= 0.5 for d in filtered)


def test_apply_nms(sample_detections: list[Detection]):
    survivors = apply_nms(sample_detections, iou_threshold=0.5)
    # raw_2 should be suppressed by raw_1
    surviving_ids = [d.object_id for d in survivors]
    assert "raw_1" in surviving_ids
    assert "raw_2" not in surviving_ids
    assert "raw_3" in surviving_ids
    assert "raw_4" in surviving_ids


def test_restore_global_coordinates():
    local_box = BBoxPx(x_min=10.0, y_min=20.0, x_max=60.0, y_max=80.0)
    global_box = restore_global_coordinates(local_box, tile_offset_x=500.0, tile_offset_y=300.0)

    assert global_box.x_min == 510.0
    assert global_box.y_min == 320.0
    assert global_box.x_max == 560.0
    assert global_box.y_max == 380.0


def test_filter_roi_center_inside(sample_detections: list[Detection]):
    # ROI covering only raw_1 and raw_2
    roi = BBoxPx(x_min=0.0, y_min=0.0, x_max=100.0, y_max=100.0)
    inside = filter_roi(sample_detections, roi=roi, policy="center_inside")
    inside_ids = [d.object_id for d in inside]

    assert "raw_1" in inside_ids
    assert "raw_2" in inside_ids
    assert "raw_3" not in inside_ids
    assert "raw_4" not in inside_ids


def test_filter_roi_intersection_ratio(sample_detections: list[Detection]):
    # Small ROI overlapping bottom right of raw_1
    roi = BBoxPx(x_min=40.0, y_min=40.0, x_max=100.0, y_max=100.0)
    # raw_1 box is 10..50 x 10..50 (area = 1600). Inter is 40..50 x 40..50 (area = 100). Ratio = 100/1600 = 0.0625
    kept_low = filter_roi(sample_detections, roi=roi, policy="intersection_over_roi_threshold", intersection_ratio_threshold=0.05)
    assert any(d.object_id == "raw_1" for d in kept_low)

    kept_high = filter_roi(sample_detections, roi=roi, policy="intersection_over_roi_threshold", intersection_ratio_threshold=0.5)
    assert not any(d.object_id == "raw_1" for d in kept_high)


def test_assign_object_ids(sample_detections: list[Detection]):
    assigned = assign_object_ids(sample_detections, prefix="aircraft")
    assert assigned[0].object_id == "aircraft_001"
    assert assigned[1].object_id == "aircraft_002"
    assert assigned[2].object_id == "aircraft_003"
    assert assigned[3].object_id == "aircraft_004"
