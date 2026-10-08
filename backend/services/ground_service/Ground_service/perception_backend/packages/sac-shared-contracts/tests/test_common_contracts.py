"""Tests for canonical common domain contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import (
    ArtifactRef,
    AssetRef,
    BBoxPx,
    Confidence,
    CoordinateFrame,
    Detection,
    EvidenceRef,
    GeoPoint,
    Modality,
    OBB,
    PointPx,
)


class TestAssetRef:
    def test_valid_asset_ref(self):
        asset = AssetRef(
            asset_id="asset_test_01",
            uri="s3://imagery/sample_optical.tif",
            modality=Modality.OPTICAL,
            width_px=1024,
            height_px=1024,
            gsd_m=0.5,
            crs="EPSG:3857",
            acquisition_time="2026-06-01T12:00:00Z",
            bands=["red", "green", "blue"],
        )
        assert asset.asset_id == "asset_test_01"
        assert asset.modality == Modality.OPTICAL
        assert asset.width_px == 1024
        assert asset.height_px == 1024
        assert asset.gsd_m == 0.5

    def test_invalid_modality_rejected(self):
        with pytest.raises(ValidationError):
            AssetRef(
                asset_id="asset_bad",
                uri="s3://imagery/bad.tif",
                modality="unsupported_quantum_band",  # type: ignore
                width_px=512,
                height_px=512,
            )

    def test_negative_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            AssetRef(
                asset_id="bad_dims",
                uri="s3://imagery/test.tif",
                modality=Modality.SAR,
                width_px=-100,
                height_px=100,
            )

    def test_extra_fields_forbidden(self):
        with pytest.raises(ValidationError):
            AssetRef(
                asset_id="extra_test",
                uri="s3://imagery/test.tif",
                modality=Modality.SAR,
                width_px=512,
                height_px=512,
                unexpected_field="disallowed",
            )


class TestBBoxPx:
    def test_valid_bbox(self):
        bbox = BBoxPx(x_min=10.0, y_min=20.0, x_max=50.0, y_max=80.0)
        assert bbox.width == 40.0
        assert bbox.height == 60.0
        assert bbox.center.x == 30.0
        assert bbox.center.y == 50.0

    def test_invalid_bbox_x_order(self):
        with pytest.raises(ValidationError, match="x_max.*>=.*x_min"):
            BBoxPx(x_min=100.0, y_min=20.0, x_max=50.0, y_max=80.0)

    def test_invalid_bbox_y_order(self):
        with pytest.raises(ValidationError, match="y_max.*>=.*y_min"):
            BBoxPx(x_min=10.0, y_min=120.0, x_max=50.0, y_max=80.0)

    def test_degenerate_zero_area_bbox_allowed(self):
        # A single point bbox (x_min == x_max, y_min == y_max) is mathematically valid
        bbox = BBoxPx(x_min=10.0, y_min=20.0, x_max=10.0, y_max=20.0)
        assert bbox.width == 0.0
        assert bbox.height == 0.0


class TestOBB:
    def test_valid_obb(self):
        obb = OBB(cx=100.0, cy=100.0, width=50.0, height=20.0, angle_deg=45.0)
        assert obb.cx == 100.0
        assert obb.angle_convention == "ccw_from_image_x"

    def test_negative_obb_dimensions_rejected(self):
        with pytest.raises(ValidationError):
            OBB(cx=100.0, cy=100.0, width=-10.0, height=20.0, angle_deg=45.0)


class TestGeoPoint:
    def test_valid_geopoint(self):
        pt = GeoPoint(latitude=28.6139, longitude=77.2090, altitude_m=216.0)
        assert pt.latitude == pytest.approx(28.6139)
        assert pt.longitude == pytest.approx(77.2090)

    def test_invalid_latitude_rejected(self):
        with pytest.raises(ValidationError):
            GeoPoint(latitude=95.0, longitude=0.0)
        with pytest.raises(ValidationError):
            GeoPoint(latitude=-91.0, longitude=0.0)

    def test_invalid_longitude_rejected(self):
        with pytest.raises(ValidationError):
            GeoPoint(latitude=0.0, longitude=185.0)
        with pytest.raises(ValidationError):
            GeoPoint(latitude=0.0, longitude=-185.0)


class TestConfidence:
    def test_valid_confidence(self):
        c = Confidence(score=0.88, calibrated=True, source="earthmind_v1")
        assert c.score == 0.88
        assert c.calibrated is True

    def test_out_of_bounds_confidence_rejected(self):
        with pytest.raises(ValidationError):
            Confidence(score=1.05, source="test")
        with pytest.raises(ValidationError):
            Confidence(score=-0.01, source="test")


class TestArtifactAndEvidence:
    def test_valid_artifact_ref(self):
        art = ArtifactRef(
            artifact_id="art_mask_01",
            artifact_type="segmentation_mask",
            uri="s3://artifacts/masks/art_mask_01.png",
            source_asset_ids=["asset_01"],
            metadata={"threshold": 0.5},
        )
        assert art.artifact_type == "segmentation_mask"

    def test_invalid_artifact_type_rejected(self):
        with pytest.raises(ValidationError):
            ArtifactRef(
                artifact_id="art_bad",
                artifact_type="arbitrary_unregistered_type",  # type: ignore
                uri="s3://artifacts/bad.bin",
            )

    def test_valid_detection_with_auto_centroid(self):
        det = Detection(
            object_id="det_01",
            class_name="storage_tank",
            confidence=Confidence(score=0.92, source="detector"),
            bbox_px=BBoxPx(x_min=0.0, y_min=0.0, x_max=10.0, y_max=20.0),
        )
        assert det.center_px is not None
        assert det.center_px.x == 5.0
        assert det.center_px.y == 10.0
