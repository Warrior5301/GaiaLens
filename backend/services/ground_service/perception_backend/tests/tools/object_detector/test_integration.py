"""Cross-workstream integration tests for Object_Detector.

Tests:
1. Team C ROI Integration: ROI_Localizer -> ROI.bbox_px -> ObjectDetectorInput.roi -> detections
2. Team B Geo Integration: Detection -> GeometryRef -> GeoMeasurement / SpatialRelationship
3. Numeric VQA Integration: can_detector_answer capability check and Mode A vs Mode B comparison
"""

from __future__ import annotations

import pytest
from app.models.registry import ToolRegistry
from app.tools.object_detector.routing import can_detector_answer
from app.tools.object_detector.tool import ObjectDetector
from evaluation.vrsbench.adapter import VRSBenchAdapter
from sac_shared_contracts.common import AssetRef, BBoxPx, Modality
from sac_shared_contracts.geo_spatial import (
    GeoMeasurementInput,
    GeometryRef,
    SpatialRelationshipInput,
    SpatialRelationType,
)
from sac_shared_contracts.perception import ObjectDetectorInput
from sac_shared_contracts.roi import ROI, ROILocalizerInput, ROILocalizerOutput


@pytest.fixture
def test_asset() -> AssetRef:
    from pathlib import Path
    img = "examples/real_satellite/05873_0000.png"
    if not Path(img).exists():
        img = "examples/sample_image.png"
    return AssetRef(
        asset_id="integration_asset_01",
        uri=img,
        modality=Modality.OPTICAL,
        width_px=512,
        height_px=512,
        gsd_m=0.5,
        crs="EPSG:3857",
    )


def test_roi_to_detector_integration(test_asset: AssetRef):
    """Verify Section 16: ROI_Localizer -> ROI.bbox_px -> ObjectDetectorInput.roi -> detections."""
    # 1. Simulate Team C ROI_Localizer output
    roi_bbox = BBoxPx(x_min=50.0, y_min=50.0, x_max=350.0, y_max=350.0)
    localized_roi = ROI(
        roi_id="roi_apron_01",
        source_asset_id=test_asset.asset_id,
        bbox_px=roi_bbox,
        reason="Aircraft parking apron",
    )
    localizer_output = ROILocalizerOutput(rois=[localized_roi])

    # 2. Extract ROI and pass directly to Object_Detector
    primary_roi = localizer_output.rois[0]
    detector_input = ObjectDetectorInput(
        asset=test_asset,
        target_classes=["airplane"],
        roi=primary_roi.bbox_px,
        confidence_threshold=0.35,
    )

    detector = ObjectDetector()
    result = detector.run(detector_input)

    assert result.status == "success"
    assert result.data is not None
    assert result.data["count"] >= 0

    # Ensure all returned detections are inside the requested ROI
    for det_dict in result.data["detections"]:
        box = det_dict["bbox_px"]
        cx = (box["x_min"] + box["x_max"]) / 2.0
        cy = (box["y_min"] + box["y_max"]) / 2.0
        assert roi_bbox.x_min <= cx <= roi_bbox.x_max
        assert roi_bbox.y_min <= cy <= roi_bbox.y_max


def test_detector_to_geo_integration(test_asset: AssetRef):
    """Verify Section 15: Detection -> GeometryRef -> Geo_Measurement / Spatial_Relationship."""
    detector = ObjectDetector()
    inp = ObjectDetectorInput(asset=test_asset, target_classes=["tennis-court"])
    res = detector.run(inp)
    assert res.status == "success"

    output = detector.validate_output(res.data)
    assert len(output.detections) > 0
    primary_det = output.detections[0]

    # Convert detection to canonical GeometryRef for Team B
    geom_ref = ObjectDetector.to_geometry_ref(primary_det)
    assert geom_ref.geometry_type == "bbox_px"
    assert geom_ref.bbox_px == primary_det.bbox_px

    # Instantiate GeoMeasurementInput with this GeometryRef
    geo_input = GeoMeasurementInput(
        asset=test_asset,
        geometry=geom_ref,
        measurement_types=["area", "perimeter"],
    )
    assert geo_input.geometry.bbox_px is not None

    # Instantiate SpatialRelationshipInput between two detections
    if len(output.detections) >= 2:
        geom_ref2 = ObjectDetector.to_geometry_ref(output.detections[1])
        spatial_input = SpatialRelationshipInput(
            asset=test_asset,
            subject=geom_ref,
            object_ref=geom_ref2,
            relation_types=[SpatialRelationType.DISTANCE],
        )
        assert spatial_input.subject.bbox_px == primary_det.bbox_px


def test_numeric_vqa_capability_routing():
    """Verify Section 18: can_detector_answer routing decisions."""
    # 1. Supported direct count query
    q1 = "How many airplanes are there?"
    assess1 = can_detector_answer(q1)
    assert assess1.detector_supported is True
    assert assess1.target_class == "airplane"
    assert assess1.requires_roi is False

    # 2. Supported count query with ROI anchor
    q2 = "How many ships are near the harbor?"
    assess2 = can_detector_answer(q2)
    assert assess2.detector_supported is True
    assert assess2.target_class == "ship"
    assert assess2.requires_roi is True
    assert assess2.roi_query == "harbor"

    # 3. Unsupported 3D/elevation query
    q3 = "What is the average height of buildings?"
    assess3 = can_detector_answer(q3)
    assert assess3.detector_supported is False

    # 4. Unsupported continuous land-cover query
    q4 = "How many distinct land-cover types are visible?"
    assess4 = can_detector_answer(q4)
    assert assess4.detector_supported is False

    # 5. Unsupported non-counting query
    q5 = "Describe the scene layout."
    assess5 = can_detector_answer(q5)
    assert assess5.detector_supported is False


def test_numeric_vqa_mode_comparison():
    """Verify Section 17: Side-by-side comparison of Mode A (VLM-only) and Mode B (Tool-backed)."""
    registry = ToolRegistry("config/tool_registry.yaml")
    adapter = VRSBenchAdapter(registry=registry)

    comparison = adapter.compare_numeric_vqa()
    assert "mode_a_vlm_only_score" in comparison
    assert "mode_b_tool_backed_score" in comparison
    assert "score_delta" in comparison
    assert comparison["sample_count"] > 0
