"""Unit tests for GroundingPipelineOrchestrator and baseline variants."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from sac_shared_contracts.common import BBoxPx
from app.core.artifacts import ArtifactStore
from app.tools.segmentation.backend import MockRemoteSAMBackend, MockSAM3Backend
from app.tools.segmentation.pipeline import (
    GroundingPipelineOrchestrator,
    extract_target_class,
)


def test_extract_target_class():
    assert extract_target_class("the large ship near the harbor") == "ship"
    assert extract_target_class("all airplanes parked on runway") == "airplane"
    assert extract_target_class("a cargo vessel docked next to the pier") == "vessel"
    assert extract_target_class("oil tank in terminal") == "oil tank"
    assert extract_target_class("the red car near the tree") == "car"


def test_pipeline_variant_remote_sam_only(tmp_path):
    img = Image.new("RGB", (200, 200), color=(20, 40, 60))
    backend = MockRemoteSAMBackend(custom_boxes=[[50, 50, 100, 100]])
    store = ArtifactStore(root_dir=str(tmp_path / "artifacts"))

    orchestrator = GroundingPipelineOrchestrator(
        remote_sam_backend=backend,
        artifact_store=store,
    )

    groundings, artifacts, meta = orchestrator.run(
        image=img,
        text_prompt="the airplane on runway",
        asset_id="test_asset_01",
        pipeline_variant="remote_sam_only",
    )

    assert len(groundings) == 1
    assert meta["pipeline"] == "remote_sam_only"
    g = groundings[0]
    assert g.class_name == "airplane"
    assert g.bbox_px.x_min == 50.0
    assert g.bbox_px.y_min == 50.0
    assert g.bbox_px.x_max == 100.0
    assert g.bbox_px.y_max == 100.0
    assert len(artifacts) >= 2


def test_pipeline_variant_remote_sam_sam3_v2(tmp_path):
    img = Image.new("RGB", (300, 300), color=(30, 30, 30))
    # Two fragmented pieces close to each other: (40-60, 40-70) and (65-85, 40-70)
    backend_r = MockRemoteSAMBackend(custom_boxes=[[40, 40, 70, 60], [40, 65, 70, 85]])
    backend_s = MockSAM3Backend()
    store = ArtifactStore(root_dir=str(tmp_path / "artifacts"))

    orchestrator = GroundingPipelineOrchestrator(
        remote_sam_backend=backend_r,
        sam3_backend=backend_s,
        artifact_store=store,
        dbscan_eps=35.0,
    )

    groundings, artifacts, meta = orchestrator.run(
        image=img,
        text_prompt="the large ship near the pier",
        asset_id="test_asset_02",
        pipeline_variant="remote_sam_sam3_v2",
        return_obb=True,
        return_polygon=True,
    )

    # DBSCAN merges the 2 fragmented pieces into 1 candidate cluster
    assert len(groundings) == 1
    g = groundings[0]
    assert g.class_name == "ship"
    assert g.obb is not None
    assert g.polygon is not None
    assert g.candidate_info["member_components"] == 2
    assert meta["pipeline"] == "remote_sam_sam3_v2"


def test_pipeline_multi_object_grounding(tmp_path):
    img = Image.new("RGB", (400, 400), color=(50, 50, 50))
    # Two distinct distant ships
    backend_r = MockRemoteSAMBackend(custom_boxes=[[30, 30, 80, 80], [250, 250, 320, 320]])
    store = ArtifactStore(root_dir=str(tmp_path / "artifacts"))

    orchestrator = GroundingPipelineOrchestrator(
        remote_sam_backend=backend_r,
        artifact_store=store,
        dbscan_eps=40.0,
    )

    groundings, artifacts, meta = orchestrator.run(
        image=img,
        text_prompt="all ships in harbor",
        asset_id="test_asset_multi",
        pipeline_variant="remote_sam_sam3_v2",
    )

    assert len(groundings) == 2
    assert meta["clusters_count"] == 2
