"""Tests for Team C ROI Localizer contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import (
    AssetRef,
    BBoxPx,
    Confidence,
    CoordinateFrame,
    Modality,
)
from sac_shared_contracts.roi import (
    ROI,
    ROILocalizerInput,
    ROILocalizerOutput,
)


@pytest.fixture
def sample_asset() -> AssetRef:
    return AssetRef(
        asset_id="roi_test_asset",
        uri="s3://imagery/roi_test.tif",
        modality=Modality.OPTICAL,
        width_px=3000,
        height_px=3000,
        gsd_m=0.3,
        crs="EPSG:3857",
    )


class TestROIContracts:
    def test_valid_roi(self):
        bbox = BBoxPx(x_min=100.0, y_min=200.0, x_max=500.0, y_max=600.0)
        roi = ROI(
            roi_id="roi_hangar_01",
            source_asset_id="asset_01",
            bbox_px=bbox,
            reason="Suspected aircraft hangar maintenance complex",
            confidence=Confidence(score=0.91, source="vlm_coarse_grounding"),
            coordinate_frame=CoordinateFrame.PIXEL,
        )
        assert roi.roi_id == "roi_hangar_01"
        assert roi.coordinate_frame == CoordinateFrame.PIXEL
        assert roi.bbox_px.width == 400.0
        assert roi.bbox_px.height == 400.0

    def test_roi_with_invalid_bbox_rejected(self):
        with pytest.raises(ValidationError):
            ROI(
                roi_id="bad_roi",
                source_asset_id="asset_01",
                bbox_px=BBoxPx(x_min=500.0, y_min=200.0, x_max=100.0, y_max=600.0),
            )

    def test_roi_localizer_input_and_output(self, sample_asset: AssetRef):
        inp = ROILocalizerInput(
            asset=sample_asset,
            target_description="Locate all aircraft dispersal bays",
            max_rois=3,
        )
        assert inp.max_rois == 3

        bbox1 = BBoxPx(x_min=100, y_min=100, x_max=400, y_max=400)
        roi1 = ROI(
            roi_id="roi_bay_01",
            source_asset_id=sample_asset.asset_id,
            bbox_px=bbox1,
            reason="Northern revetment bay",
        )
        out = ROILocalizerOutput(rois=[roi1])
        assert len(out.rois) == 1
