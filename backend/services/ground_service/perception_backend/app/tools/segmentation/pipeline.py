"""Grounding pipeline orchestrator supporting RemoteSAM, v1, and v2 (DBSCAN + SAM3)."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any, Optional
import numpy as np
from PIL import Image
from sac_shared_contracts.common import (
    ArtifactRef,
    BBoxPx,
    Confidence,
    OBB,
)
from sac_shared_contracts.perception import GroundingResult

from app.core.artifacts import ArtifactStore
from app.tools.segmentation.backend import (
    RemoteSAMBackend,
    SAM3Backend,
)
from app.tools.segmentation.candidate import (
    CandidateScore,
    DBSCANCluster,
    GroundingCandidate,
)
from app.tools.segmentation.clustering import (
    cluster_components_dbscan,
    extract_connected_components,
)
from app.tools.segmentation.geometry import (
    mask_to_aabb,
    mask_to_obb,
    mask_to_polygon,
)
from app.tools.segmentation.postprocess import (
    cleanup_binary_mask,
    logits_to_binary_mask,
)
from app.tools.segmentation.remote_sam import RemoteSAMWrapper
from app.tools.segmentation.sam3 import SAM3Wrapper
from app.tools.segmentation.selector import CandidateSelector

logger = logging.getLogger("app.tools.segmentation.pipeline")

# Common remote-sensing target nouns for semantic target extraction
COMMON_TARGET_CLASSES = [
    "airplane", "aircraft", "plane", "jet",
    "ship", "vessel", "boat", "warship", "cargo ship", "carrier",
    "vehicle", "car", "truck", "bus",
    "building", "hangar", "warehouse", "terminal",
    "storage tank", "oil tank", "fuel tank", "tank",
    "bridge", "pier", "dock", "wharf", "runway", "helipad",
    "solar panel", "wind turbine", "crane", "container",
]


def extract_target_class(referring_expression: str) -> str:
    """Extract primary semantic target class from a natural language referring expression.

    Separate full referring expression ('the large red ship near the harbor')
    from semantic target class ('ship').
    """
    clean_expr = referring_expression.lower().strip()

    # 1. Match known common remote sensing classes, prioritizing the first-mentioned noun
    matches: list[tuple[int, str]] = []
    for cls_name in sorted(COMMON_TARGET_CLASSES, key=len, reverse=True):
        pattern = r"\b" + re.escape(cls_name) + r"(?:s|es)?\b"
        match = re.search(pattern, clean_expr)
        if match:
            matches.append((match.start(), cls_name))

    if matches:
        # Pick the match that appears earliest in the sentence (the subject)
        matches.sort(key=lambda m: m[0])
        return matches[0][1]

    # 2. Heuristic fallback: remove common stop/modifier words
    words = re.findall(r"\b[a-zA-Z]+\b", clean_expr)
    stop_words = {
        "the", "a", "an", "all", "each", "every", "this", "that", "those", "these",
        "large", "small", "huge", "tiny", "medium", "big", "little",
        "red", "white", "black", "blue", "gray", "grey", "green", "yellow",
        "near", "next", "to", "beside", "adjacent", "on", "in", "at", "by", "over",
        "runway", "pier", "harbor", "airport", "water", "sea", "ground", "grass",
        "left", "right", "top", "bottom", "north", "south", "east", "west",
    }
    candidate_words = [w for w in words if w not in stop_words]
    if candidate_words:
        # Return the last substantive noun
        return candidate_words[-1]

    return "target_object"


class GroundingPipelineOrchestrator:
    """Coordinates RemoteSAM, connected components, DBSCAN, SAM3, and geometry extraction."""

    def __init__(
        self,
        remote_sam_backend: Optional[RemoteSAMBackend] = None,
        sam3_backend: Optional[SAM3Backend] = None,
        artifact_store: Optional[ArtifactStore] = None,
        dbscan_eps: float = 40.0,
        dbscan_min_samples: int = 1,
        logit_threshold: float = 0.5,
        min_component_area: int = 10,
        candidate_selector: Optional[CandidateSelector] = None,
    ) -> None:
        self.remote_sam = RemoteSAMWrapper(backend=remote_sam_backend)
        self.sam3 = SAM3Wrapper(backend=sam3_backend)
        self.artifact_store = artifact_store or ArtifactStore()
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.logit_threshold = logit_threshold
        self.min_component_area = min_component_area
        self.selector = candidate_selector or CandidateSelector()

    def run(
        self,
        image: Image.Image,
        text_prompt: str,
        asset_id: str,
        pipeline_variant: str = "remote_sam_sam3_v2",
        roi: Optional[BBoxPx] = None,
        return_polygon: bool = True,
        return_obb: bool = True,
        confidence_threshold: float = 0.35,
    ) -> tuple[list[GroundingResult], list[ArtifactRef], dict[str, Any]]:
        """Execute the visual grounding pipeline.

        Returns:
            Tuple of (groundings, artifacts, pipeline_metadata).
        """
        start_time = time.perf_counter()
        w, h = image.size
        artifacts: list[ArtifactRef] = []

        # Canonical pipeline variant normalization
        variant = pipeline_variant.lower()
        if variant == "remote_sam":
            variant = "remote_sam_only"

        # 1. Semantic target class extraction
        target_class = extract_target_class(text_prompt)

        # 2. Stage 1: RemoteSAM inference
        logits, rsam_meta = self.remote_sam.run_grounding(image, text_prompt=text_prompt, roi=roi)

        # Save raw logit / prob mask artifact
        prob_mask = 1.0 / (1.0 + np.exp(-np.clip(logits, -50.0, 50.0)))
        logit_art = self.artifact_store.save_mask(
            (prob_mask * 255).astype(np.uint8),
            source_asset_id=asset_id,
            metadata={"type": "logit_probability_map", "prompt": text_prompt},
        )
        artifacts.append(logit_art)

        # 3. Thresholding & Morphological Cleanup
        raw_binary = logits_to_binary_mask(logits, threshold=self.logit_threshold, apply_sigmoid=True)
        cleaned_binary = cleanup_binary_mask(
            raw_binary,
            min_area_pixels=self.min_component_area,
            fill_holes=True,
        )

        binary_art = self.artifact_store.save_mask(
            cleaned_binary,
            source_asset_id=asset_id,
            metadata={"type": "thresholded_binary_mask", "threshold": self.logit_threshold},
        )
        artifacts.append(binary_art)

        # 4. Connected Components Extraction
        components = extract_connected_components(
            cleaned_binary,
            min_area_pixels=self.min_component_area,
        )

        if not components:
            logger.info("No foreground components found above min_area=%d", self.min_component_area)
            return [], artifacts, {
                "pipeline": variant,
                "target_class": target_class,
                "components_found": 0,
                "clusters_found": 0,
                "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
            }

        # 5. Pipeline Branching
        candidate_clusters: list[DBSCANCluster] = []
        if variant == "remote_sam_only":
            # RemoteSAM only: each connected component directly becomes a cluster
            for idx, comp in enumerate(components, start=1):
                candidate_clusters.append(
                    DBSCANCluster(
                        cluster_id=idx,
                        member_components=[comp],
                        mask=comp.mask.copy(),
                        bbox_px=comp.bbox_px,
                        area_pixels=comp.area_pixels,
                        centroid=comp.centroid,
                    )
                )
        elif variant == "remote_sam_sam3_v1":
            # v1: Direct connected components to SAM3 without DBSCAN
            for idx, comp in enumerate(components, start=1):
                candidate_clusters.append(
                    DBSCANCluster(
                        cluster_id=idx,
                        member_components=[comp],
                        mask=comp.mask.copy(),
                        bbox_px=comp.bbox_px,
                        area_pixels=comp.area_pixels,
                        centroid=comp.centroid,
                    )
                )
        else:  # "remote_sam_sam3_v2" (default)
            # v2: DBSCAN clustering of fragmented components into coherent regions
            candidate_clusters = cluster_components_dbscan(
                components,
                eps=self.dbscan_eps,
                min_samples=self.dbscan_min_samples,
                image_shape=(h, w),
            )

        # 6. Candidate Refinement & Selection Loop
        groundings: list[GroundingResult] = []
        all_candidate_diagnostics: list[dict[str, Any]] = []

        for cluster in candidate_clusters:
            if variant == "remote_sam_only":
                # RemoteSAM only: use cluster mask directly as winner
                winner_mask = cluster.mask
                winning_score = CandidateScore(
                    iou_with_cluster=1.0,
                    sam3_confidence=0.85,
                    final_score=0.85,
                )
                source_model_id = self.remote_sam.backend.model_id
                diag = {"pipeline": "remote_sam_only", "selected": "remotesam_component"}
            else:
                # SAM3 refinement
                raw_candidates = self.sam3.refine_candidate_cluster(
                    image=image,
                    object_name=target_class,
                    cluster_mask=cluster.mask,
                    cluster_bbox=cluster.bbox_px,
                    num_candidates=3,
                )

                evaluated_candidates: list[GroundingCandidate] = []
                for c_idx, (cand_mask, sam3_conf, meta) in enumerate(raw_candidates, start=1):
                    cand_score = self.selector.score_candidate(
                        candidate_mask=cand_mask,
                        cluster=cluster,
                        sam3_confidence=sam3_conf,
                    )
                    evaluated_candidates.append(
                        GroundingCandidate(
                            candidate_id=f"cluster_{cluster.cluster_id}_cand_{c_idx}",
                            cluster_id=cluster.cluster_id,
                            mask=cand_mask,
                            score=cand_score,
                            source_model=self.sam3.backend.model_id,
                            metadata=meta,
                        )
                    )

                # Select best candidate
                selected_cand, diag = self.selector.select_best_candidate(
                    candidates=evaluated_candidates,
                    fallback_cluster=cluster,
                )

                if selected_cand is None:
                    continue

                winner_mask = selected_cand.mask
                winning_score = selected_cand.score
                source_model_id = selected_cand.source_model

            all_candidate_diagnostics.append(diag)

            # Filter by confidence threshold
            conf_val = float(winning_score.sam3_confidence or winning_score.final_score)
            if conf_val < confidence_threshold:
                continue

            # 7. Geometric derivation (AABB, OBB, Polygon)
            aabb = mask_to_aabb(winner_mask)
            obb = mask_to_obb(winner_mask) if return_obb else None
            polygon = mask_to_polygon(winner_mask, tolerance=1.5) if return_polygon else None

            # Save final mask artifact
            final_mask_art = self.artifact_store.save_mask(
                winner_mask.astype(np.uint8),
                source_asset_id=asset_id,
                metadata={
                    "type": "grounding_mask",
                    "object_class": target_class,
                    "cluster_id": cluster.cluster_id,
                    "pipeline": variant,
                },
            )
            artifacts.append(final_mask_art)

            grounding_id = f"ground_{uuid.uuid4().hex[:8]}"
            groundings.append(
                GroundingResult(
                    object_id=grounding_id,
                    class_name=target_class,
                    referring_target=text_prompt,
                    confidence=Confidence(
                        score=round(conf_val, 4),
                        calibrated=False,
                        source=f"{variant}:{source_model_id}",
                    ),
                    bbox_px=aabb,
                    obb=obb,
                    polygon=polygon,
                    mask_artifact=final_mask_art,
                    source_model=f"{variant}:{source_model_id}",
                    candidate_info={
                        "cluster_id": cluster.cluster_id,
                        "cluster_area_px": cluster.area_pixels,
                        "member_components": len(cluster.member_components),
                        "iou_with_cluster": winning_score.iou_with_cluster,
                        "diagnostics": diag,
                    },
                )
            )

        total_latency_ms = round((time.perf_counter() - start_time) * 1000.0, 2)
        pipeline_meta = {
            "pipeline": variant,
            "target_class": target_class,
            "referring_expression": text_prompt,
            "components_count": len(components),
            "clusters_count": len(candidate_clusters),
            "groundings_count": len(groundings),
            "dbscan_eps": self.dbscan_eps,
            "dbscan_min_samples": self.dbscan_min_samples,
            "logit_threshold": self.logit_threshold,
            "latency_ms": total_latency_ms,
            "candidate_diagnostics": all_candidate_diagnostics,
        }

        return groundings, artifacts, pipeline_meta
