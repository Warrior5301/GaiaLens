"""RemoteSAM model wrapper for text-guided promptable remote sensing segmentation."""

from __future__ import annotations

import logging
import time
from typing import Any, Optional
import numpy as np
from PIL import Image
from sac_shared_contracts.common import BBoxPx
from app.tools.segmentation.backend import RemoteSAMBackend

logger = logging.getLogger("app.tools.segmentation.remote_sam")


class RemoteSAMWrapper:
    """High-level wrapper managing RemoteSAM inference and ROI sub-region extraction."""

    def __init__(self, backend: Optional[RemoteSAMBackend] = None) -> None:
        self.backend = backend or RemoteSAMBackend()

    def run_grounding(
        self,
        image: Image.Image,
        text_prompt: str,
        roi: Optional[BBoxPx] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Run text-guided visual grounding on full image or ROI sub-region.

        Returns:
            Tuple of (full_image_logits: np.ndarray, execution_metadata: dict).
        """
        start_time = time.perf_counter()
        w, h = image.size

        if roi is not None:
            # Crop image to ROI for regional inference
            crop_box = (
                max(0, int(roi.x_min)),
                max(0, int(roi.y_min)),
                min(w, int(roi.x_max)),
                min(h, int(roi.y_max)),
            )
            cropped_image = image.crop(crop_box)
            crop_w, crop_h = cropped_image.size

            if crop_w > 0 and crop_h > 0:
                crop_logits = self.backend.predict_logits(cropped_image, text_prompt)
                # Embed back into full-size logit canvas
                full_logits = -10.0 * np.ones((h, w), dtype=np.float32)
                full_logits[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]] = crop_logits
            else:
                full_logits = self.backend.predict_logits(image, text_prompt)
        else:
            full_logits = self.backend.predict_logits(image, text_prompt)

        duration_ms = (time.perf_counter() - start_time) * 1000.0
        metadata = {
            "model_id": self.backend.model_id,
            "architecture": self.backend.ARCH_NAME,
            "latency_ms": round(duration_ms, 2),
            "image_shape": [h, w],
            "text_prompt": text_prompt,
            "roi_applied": roi is not None,
        }

        return full_logits, metadata
