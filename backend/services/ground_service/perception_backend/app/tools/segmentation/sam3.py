"""SAM3 promptable refinement wrapper with crop-handling and candidate generation."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
from sac_shared_contracts.common import BBoxPx
from app.tools.segmentation.backend import SAM3Backend

logger = logging.getLogger("app.tools.segmentation.sam3")


class SAM3Wrapper:
    """Wrapper managing SAM3 candidate mask refinement with adaptive local cropping."""

    def __init__(
        self,
        backend: Optional[SAM3Backend] = None,
        use_crop_refinement: bool = True,
        crop_padding_px: int = 20,
    ) -> None:
        self.backend = backend or SAM3Backend()
        self.use_crop_refinement = use_crop_refinement
        self.crop_padding_px = crop_padding_px

    def refine_candidate_cluster(
        self,
        image: Image.Image,
        object_name: str,
        cluster_mask: np.ndarray,
        cluster_bbox: BBoxPx,
        num_candidates: int = 3,
    ) -> list[tuple[np.ndarray, float, dict[str, Any]]]:
        """Refine candidate cluster into SAM3 hypotheses.

        Returns:
            List of (full_size_candidate_mask: np.ndarray, confidence: float, metadata: dict).
        """
        start_time = time.perf_counter()
        img_w, img_h = image.size
        mask_h, mask_w = cluster_mask.shape

        if self.use_crop_refinement:
            # Crop image and cluster mask around the cluster bounding box
            pad = self.crop_padding_px
            x1 = max(0, int(cluster_bbox.x_min - pad))
            y1 = max(0, int(cluster_bbox.y_min - pad))
            x2 = min(img_w, int(cluster_bbox.x_max + pad))
            y2 = min(img_h, int(cluster_bbox.y_max + pad))

            crop_w = x2 - x1
            crop_h = y2 - y1

            if crop_w > 5 and crop_h > 5:
                crop_img = image.crop((x1, y1, x2, y2))
                crop_cluster_mask = cluster_mask[y1:y2, x1:x2]

                raw_candidates = self.backend.refine_cluster(
                    crop_img,
                    object_name=object_name,
                    cluster_mask=crop_cluster_mask,
                    num_candidates=num_candidates,
                )

                refined_results: list[tuple[np.ndarray, float, dict[str, Any]]] = []
                for cand_mask, conf in raw_candidates:
                    # Embed candidate mask back into full canvas
                    full_cand = np.zeros((img_h, img_w), dtype=np.uint8)
                    full_cand[y1:y2, x1:x2] = cand_mask
                    meta = {
                        "crop_bbox": [x1, y1, x2, y2],
                        "model_id": self.backend.model_id,
                        "latency_ms": round((time.perf_counter() - start_time) * 1000.0, 2),
                    }
                    refined_results.append((full_cand, conf, meta))

                return refined_results

        # Full image inference without cropping
        raw_candidates = self.backend.refine_cluster(
            image,
            object_name=object_name,
            cluster_mask=cluster_mask,
            num_candidates=num_candidates,
        )
        duration_ms = (time.perf_counter() - start_time) * 1000.0
        return [
            (
                cand_mask,
                conf,
                {
                    "crop_applied": False,
                    "model_id": self.backend.model_id,
                    "latency_ms": round(duration_ms, 2),
                },
            )
            for cand_mask, conf in raw_candidates
        ]
