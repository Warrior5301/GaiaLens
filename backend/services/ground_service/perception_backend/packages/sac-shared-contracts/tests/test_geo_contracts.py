"""Tests for Team B Geo/Spatial contracts and semantic validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import (
    AssetRef,
    BBoxPx,
    Confidence,
    GeoPoint,
    Modality,
    PointPx,
)
from sac_shared_contracts.geo_spatial import (
    GeoMeasurementInput,
    GeoMeasurementOutput,
    GeometryRef,
    MeasurementValue,
    SpatialRelationship,
    SpatialRelationshipInput,
    SpatialRelationshipOutput,
    SpatialRelationType,
)


@pytest.fixture
def sample_asset() -> AssetRef:
    return AssetRef(
        asset_id="geo_test_asset",
        uri="s3://imagery/geo_test.tif",
        modality=Modality.OPTICAL,
        width_px=2000,
        height_px=2000,
        gsd_m=0.5,
        crs="EPSG:32643",
    )


class TestGeometryRefDiscrimination:
    def test_valid_bbox_ref(self):
        bbox = BBoxPx(x_min=10, y_min=10, x_max=50, y_max=60)
        geom = GeometryRef(geometry_type="bbox_px", bbox_px=bbox)
        assert geom.geometry_type == "bbox_px"
        assert geom.bbox_px == bbox

    def test_missing_bbox_px_fails(self):
        with pytest.raises(ValidationError, match="requires 'bbox_px' field"):
            GeometryRef(geometry_type="bbox_px", bbox_px=None)

    def test_valid_polygon_ref(self):
        poly = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 0.0]]
        geom = GeometryRef(geometry_type="polygon", polygon=poly)
        assert geom.geometry_type == "polygon"
        assert len(geom.polygon) == 4

    def test_polygon_fewer_than_three_vertices_fails(self):
        with pytest.raises(ValidationError, match="at least 3 vertices"):
            GeometryRef(geometry_type="polygon", polygon=[[0.0, 0.0], [1.0, 1.0]])

    def test_valid_point_ref(self):
        pt = PointPx(x=15.0, y=25.0)
        geom = GeometryRef(geometry_type="point", point=pt)
        assert geom.geometry_type == "point"
        assert geom.point == pt

    def test_missing_point_fails(self):
        with pytest.raises(ValidationError, match="requires 'point' field"):
            GeometryRef(geometry_type="point", point=None)

    def test_valid_artifact_ref(self):
        geom = GeometryRef(geometry_type="artifact", artifact_id="mask_runway_01")
        assert geom.geometry_type == "artifact"
        assert geom.artifact_id == "mask_runway_01"

    def test_missing_artifact_id_fails(self):
        with pytest.raises(ValidationError, match="requires 'artifact_id' field"):
            GeometryRef(geometry_type="artifact", artifact_id=None)


class TestGeoMeasurement:
    def test_valid_measurement_input_and_output(self, sample_asset: AssetRef):
        bbox = BBoxPx(x_min=100, y_min=100, x_max=300, y_max=500)
        geom = GeometryRef(geometry_type="bbox_px", bbox_px=bbox)

        inp = GeoMeasurementInput(
            asset=sample_asset,
            geometry=geom,
            measurement_types=["area", "perimeter"],
        )
        assert inp.asset.asset_id == sample_asset.asset_id
        assert inp.measurement_types == ["area", "perimeter"]

        out = GeoMeasurementOutput(
            measurements={
                "area": MeasurementValue(
                    value=20000.0,
                    unit="m2",
                    confidence=Confidence(score=0.95, source="gsd_formula"),
                    method="pixel_gsd",
                ),
                "perimeter": MeasurementValue(
                    value=600.0,
                    unit="m",
                    confidence=Confidence(score=0.92, source="gsd_formula"),
                    method="pixel_gsd",
                ),
            }
        )
        assert out.measurements["area"].value == 20000.0
        assert out.measurements["perimeter"].unit == "m"


class TestSpatialRelationship:
    def test_spatial_relationship_evaluation(self, sample_asset: AssetRef):
        geom_a = GeometryRef(
            geometry_type="bbox_px",
            bbox_px=BBoxPx(x_min=10, y_min=10, x_max=50, y_max=50),
        )
        geom_b = GeometryRef(
            geometry_type="bbox_px",
            bbox_px=BBoxPx(x_min=100, y_min=100, x_max=200, y_max=200),
        )

        inp = SpatialRelationshipInput(
            asset=sample_asset,
            subject=geom_a,
            object_ref=geom_b,
            relation_types=[SpatialRelationType.DISTANCE],
        )
        assert inp.relation_types == [SpatialRelationType.DISTANCE]

        rel = SpatialRelationship(
            relation_type=SpatialRelationType.DISTANCE,
            satisfied=True,
            confidence=Confidence(score=0.98, source="cartesian_distance"),
            distance_m=70.7,
        )
        out = SpatialRelationshipOutput(relationships=[rel])
        assert out.relationships[0].distance_m == 70.7
        assert out.relationships[0].satisfied is True
