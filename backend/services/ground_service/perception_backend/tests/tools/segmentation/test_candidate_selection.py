"""Unit tests for candidate scoring, ranking, and fallback selection."""

from __future__ import annotations

import numpy as np
import pytest
from sac_shared_contracts.common import BBoxPx
from app.tools.segmentation.candidate import (
    CandidateRegion,
    CandidateScore,
    DBSCANCluster,
    GroundingCandidate,
)
from app.tools.segmentation.selector import CandidateSelector


def test_candidate_selector_chooses_highest_iou():
    selector = CandidateSelector(min_iou_threshold=0.1)

    # Reference cluster mask
    cluster_mask = np.zeros((50, 50), dtype=bool)
    cluster_mask[10:30, 10:30] = True

    cluster = DBSCANCluster(
        cluster_id=1,
        member_components=[],
        mask=cluster_mask,
        bbox_px=BBoxPx(x_min=10.0, y_min=10.0, x_max=30.0, y_max=30.0),
        area_pixels=400,
        centroid=(20.0, 20.0),
    )

    # Candidate 1: High overlap (90% IoU)
    cand1_mask = cluster_mask.copy()
    cand1_mask[10:12, 10:30] = False  # Slightly smaller
    cand1 = GroundingCandidate(
        candidate_id="cand_high",
        cluster_id=1,
        mask=cand1_mask,
        score=selector.score_candidate(cand1_mask, cluster, sam3_confidence=0.9),
        source_model="sam3",
    )

    # Candidate 2: Low overlap
    cand2_mask = np.zeros((50, 50), dtype=bool)
    cand2_mask[25:35, 25:35] = True
    cand2 = GroundingCandidate(
        candidate_id="cand_low",
        cluster_id=1,
        mask=cand2_mask,
        score=selector.score_candidate(cand2_mask, cluster, sam3_confidence=0.6),
        source_model="sam3",
    )

    winner, diag = selector.select_best_candidate([cand2, cand1], fallback_cluster=cluster)
    assert winner is not None
    assert winner.candidate_id == "cand_high"
    assert winner.score.iou_with_cluster > 0.8
    assert diag["fallback_used"] is False


def test_candidate_selector_fallback_on_empty():
    selector = CandidateSelector()

    cluster_mask = np.ones((10, 10), dtype=bool)
    cluster = DBSCANCluster(
        cluster_id=2,
        member_components=[],
        mask=cluster_mask,
        bbox_px=BBoxPx(x_min=0.0, y_min=0.0, x_max=10.0, y_max=10.0),
        area_pixels=100,
        centroid=(5.0, 5.0),
    )

    winner, diag = selector.select_best_candidate([], fallback_cluster=cluster)
    assert winner is not None
    assert diag["fallback_used"] is True
    assert winner.candidate_id == "cluster_2_fallback"
    assert np.array_equal(winner.mask, cluster_mask)
