"""Tests for Team C OCR extraction contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import AssetRef, BBoxPx, Confidence, Modality
from sac_shared_contracts.ocr import OCRInput, OCROutput, OCRTextRegion


@pytest.fixture
def ocr_asset() -> AssetRef:
    return AssetRef(
        asset_id="airfield_asset_01",
        uri="s3://imagery/runway_marking.tif",
        modality=Modality.OPTICAL,
        width_px=1024,
        height_px=1024,
        gsd_m=0.15,
        crs="EPSG:3857",
    )


class TestOCRContracts:
    def test_valid_ocr_text_region(self):
        bbox = BBoxPx(x_min=100.0, y_min=200.0, x_max=250.0, y_max=240.0)
        region = OCRTextRegion(
            text_id="txt_01",
            text="27R",
            bbox_px=bbox,
            confidence=Confidence(score=0.98, source="tesseract_crnn"),
            language="en",
            rotation_deg=90.0,
        )
        assert region.text == "27R"
        assert region.bbox_px.width == 150.0
        assert region.rotation_deg == 90.0

    def test_ocr_input_and_output(self, ocr_asset: AssetRef):
        roi_bbox = BBoxPx(x_min=50.0, y_min=50.0, x_max=400.0, y_max=400.0)
        inp = OCRInput(
            asset=ocr_asset,
            roi=roi_bbox,
            languages=["en"],
        )
        assert inp.languages == ["en"]
        assert inp.roi is not None
        assert inp.roi.width == 350.0

        region = OCRTextRegion(
            text_id="txt_01",
            text="09L",
            bbox_px=BBoxPx(x_min=60.0, y_min=80.0, x_max=120.0, y_max=110.0),
            confidence=Confidence(score=0.97, source="ocr_model"),
        )
        out = OCROutput(
            regions=[region],
            full_text="09L",
            metadata={"tiles_scanned": 1},
        )
        assert len(out.regions) == 1
        assert out.full_text == "09L"
