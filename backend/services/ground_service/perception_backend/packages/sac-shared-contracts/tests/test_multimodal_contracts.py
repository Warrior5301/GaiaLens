"""Tests for Team C Multimodal Fusion and Sensor Alignment contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import ArtifactRef, AssetRef, Modality
from sac_shared_contracts.multimodal import (
    AlignmentReport,
    FusionInput,
    FusionOutput,
    ModalityAsset,
    ModalityRole,
)


@pytest.fixture
def optical_asset() -> AssetRef:
    return AssetRef(
        asset_id="opt_ref_01",
        uri="s3://imagery/optical_base.tif",
        modality=Modality.OPTICAL,
        width_px=2048,
        height_px=2048,
        gsd_m=0.5,
        crs="EPSG:3857",
    )


@pytest.fixture
def sar_asset() -> AssetRef:
    return AssetRef(
        asset_id="sar_sec_01",
        uri="s3://imagery/sar_layer.tif",
        modality=Modality.SAR,
        width_px=1024,
        height_px=1024,
        gsd_m=1.0,
        crs="EPSG:4326",
    )


class TestMultimodalContracts:
    def test_valid_modality_asset(self, sar_asset: AssetRef):
        mod_asset = ModalityAsset(asset=sar_asset, role=ModalityRole.SAR)
        assert mod_asset.role == ModalityRole.SAR
        assert mod_asset.asset.asset_id == "sar_sec_01"

    def test_valid_alignment_report(self):
        report = AlignmentReport(
            registered=True,
            residual_rmse_px=0.42,
            target_crs="EPSG:3857",
            target_resolution_m=0.5,
            notes="Homography coregistration using 28 tie-points",
        )
        assert report.registered is True
        assert report.residual_rmse_px == 0.42

    def test_invalid_negative_rmse_rejected(self):
        with pytest.raises(ValidationError):
            AlignmentReport(
                registered=True,
                residual_rmse_px=-1.5,
                target_crs="EPSG:3857",
                target_resolution_m=0.5,
            )

    def test_invalid_zero_resolution_rejected(self):
        with pytest.raises(ValidationError):
            AlignmentReport(
                registered=True,
                residual_rmse_px=0.5,
                target_crs="EPSG:3857",
                target_resolution_m=0.0,
            )

    def test_valid_fusion_flow(self, optical_asset: AssetRef, sar_asset: AssetRef):
        inp = FusionInput(
            reference_asset=optical_asset,
            secondary_assets=[
                ModalityAsset(asset=sar_asset, role=ModalityRole.SAR)
            ],
            target_crs="EPSG:3857",
            target_resolution_m=0.5,
        )
        assert inp.reference_asset.asset_id == optical_asset.asset_id
        assert len(inp.secondary_assets) == 1

        out = FusionOutput(
            fused_artifact=ArtifactRef(
                artifact_id="fused_opt_sar_01",
                artifact_type="aligned_asset",
                uri="s3://artifacts/fused/opt_sar_01.tif",
                source_asset_ids=[optical_asset.asset_id, sar_asset.asset_id],
            ),
            alignment_report=AlignmentReport(
                registered=True,
                residual_rmse_px=0.35,
                target_crs="EPSG:3857",
                target_resolution_m=0.5,
            ),
        )
        assert out.fused_artifact.artifact_type == "aligned_asset"
        assert out.alignment_report.registered is True
