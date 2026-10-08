"""Candidate scoring and selection algorithms for SAM3 refinement."""

from __future__ import annotations

import logging
from typing import Optional
import numpy as np
from app.tools.segmentation.candidate import (
    CandidateScore,
    DBSCANCluster,
    GroundingCandidate,
)
from app.tools.segmentation.geometry import compute_mask_iou, mask_to_aabb

logger = logging.getLogger("app.tools.segmentation.selector")


class CandidateSelector:
    """Evaluates and selects the optimal mask candidate for each candidate cluster.

    Follows the paper's RemoteSAM-SAM3-v2 baseline criterion:
    The primary candidate selection score is the IoU between the SAM3 candidate mask
    and the originating DBSCAN cluster mask.
    """

    def __init__(
        self,
        min_iou_threshold: float = 0.1,
        alpha_iou: float = 1.0,
    ) -> None:
        self.min_iou_threshold = min_iou_threshold
        self.alpha_iou = alpha_iou

    def score_candidate(
        self,
        candidate_mask: np.ndarray,
        cluster: DBSCANCluster,
        sam3_confidence: float = 0.0,
        semantic_confidence: float = 1.0,
    ) -> CandidateScore:
        """Compute candidate score against cluster reference mask."""
        iou = compute_mask_iou(candidate_mask, cluster.mask)
        cand_area = float(np.sum(candidate_mask > 0))
        cluster_area = float(cluster.area_pixels)

        # Area consistency ratio (1.0 if identical, lower if drastically divergent)
        if cand_area > 0 and cluster_area > 0:
            area_ratio = min(cand_area, cluster_area) / max(cand_area, cluster_area)
        else:
            area_ratio = 0.0

        score = CandidateScore(
            iou_with_cluster=round(float(iou), 4),
            sam3_confidence=round(float(sam3_confidence), 4),
            area_consistency=round(float(area_ratio), 4),
            semantic_confidence=round(float(semantic_confidence), 4),
        )
        score.compute_total_score(alpha_iou=self.alpha_iou)
        return score

    def select_best_candidate(
        self,
        candidates: list[GroundingCandidate],
        fallback_cluster: Optional[DBSCANCluster] = None,
    ) -> tuple[Optional[GroundingCandidate], dict[str, any]]:
        """Select the highest scoring candidate from a pool of SAM3 candidates for a cluster.

        Args:
            candidates: Evaluated GroundingCandidate instances.
            fallback_cluster: If all candidates fail or score below threshold, fall back to cluster mask.

        Returns:
            Tuple of (selected GroundingCandidate, candidate diagnostics dictionary).
        """
        diagnostics: dict[str, any] = {
            "num_candidates_evaluated": len(candidates),
            "candidate_scores": [
                {
                    "candidate_id": c.candidate_id,
                    "iou_with_cluster": c.score.iou_with_cluster,
                    "sam3_confidence": c.score.sam3_confidence,
                    "final_score": c.score.final_score,
                }
                for c in candidates
            ],
            "fallback_used": False,
        }

        if not candidates:
            if fallback_cluster is not None:
                logger.info(
                    "No SAM3 candidates produced for cluster %d; falling back to cluster mask",
                    fallback_cluster.cluster_id,
                )
                fallback_cand = GroundingCandidate(
                    candidate_id=f"cluster_{fallback_cluster.cluster_id}_fallback",
                    cluster_id=fallback_cluster.cluster_id,
                    mask=fallback_cluster.mask.copy(),
                    score=CandidateScore(
                        iou_with_cluster=1.0,
                        sam3_confidence=0.5,
                        final_score=1.0,
                    ),
                    source_model="dbscan_cluster_fallback",
                    bbox_px=fallback_cluster.bbox_px,
                )
                diagnostics["fallback_used"] = True
                diagnostics["selected_id"] = fallback_cand.candidate_id
                return fallback_cand, diagnostics
            return None, diagnostics

        # Sort descending by final score
        sorted_candidates = sorted(candidates, key=lambda c: c.score.final_score, reverse=True)
        winner = sorted_candidates[0]

        # Check if winner meets minimum IoU threshold
        if winner.score.iou_with_cluster < self.min_iou_threshold and fallback_cluster is not None:
            logger.info(
                "Winner IoU %f below threshold %f for cluster %d; using cluster mask fallback",
                winner.score.iou_with_cluster,
                self.min_iou_threshold,
                fallback_cluster.cluster_id,
            )
            fallback_cand = GroundingCandidate(
                candidate_id=f"cluster_{fallback_cluster.cluster_id}_fallback",
                cluster_id=fallback_cluster.cluster_id,
                mask=fallback_cluster.mask.copy(),
                score=CandidateScore(
                    iou_with_cluster=1.0,
                    sam3_confidence=0.5,
                    final_score=1.0,
                ),
                source_model="dbscan_cluster_fallback",
                bbox_px=fallback_cluster.bbox_px,
            )
            diagnostics["fallback_used"] = True
            diagnostics["selected_id"] = fallback_cand.candidate_id
            return fallback_cand, diagnostics

        diagnostics["selected_id"] = winner.candidate_id
        diagnostics["winning_score"] = winner.score.final_score
        return winner, diagnostics
