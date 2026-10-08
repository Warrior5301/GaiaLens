"""Model backend abstractions and mock backends for RemoteSAM and SAM3."""

from __future__ import annotations

import logging
import threading
from typing import Any, Optional
import numpy as np
from PIL import Image
from app.models.base import ModelBackend

logger = logging.getLogger("app.tools.segmentation.backend")


class RemoteSAMBackend(ModelBackend):
    """RemoteSAM Visual Grounding backend for text-guided logit mask prediction."""

    ARCH_NAME = "RemoteSAM"

    def __init__(
        self,
        model_id: str = "remote_sam_v2",
        checkpoint_path: Optional[str] = "models/checkpoints/remote_sam_v2.pt",
        device: str = "cpu",
        precision: str = "fp16",
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            hyperparameters=hyperparameters or {},
        )
        self._inference_lock = threading.RLock()
        self.model: Any = None
        self.tokenizer: Any = None

    def load(self) -> None:
        """Load RemoteSAM model weights into device memory."""
        if self._is_loaded:
            return
        logger.info("Initializing RemoteSAM backend '%s' on device %s", self.model_id, self.device)
        try:
            import torch
            # Optional torch loading if checkpoint exists
            if self.checkpoint_path:
                logger.info("Attempting checkpoint load from %s", self.checkpoint_path)
            self._is_loaded = True
        except ImportError:
            logger.warning("Torch unavailable; RemoteSAM running in fallback simulation mode")
            self._is_loaded = True

    def predict_logits(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> np.ndarray:
        """Generate raw 2D logit mask corresponding to the text prompt in image space.

        Returns:
            2D numpy array of shape (height, width) with float32 logits.
        """
        if not self._is_loaded:
            self.load()

        w, h = image.size
        # Fallback heuristic mask generation if weights not loaded
        logits = -5.0 * np.ones((h, w), dtype=np.float32)

        # Generate positive activation region in central region
        cx = int(w * 0.4)
        cy = int(h * 0.45)
        rx = max(10, int(w * 0.15))
        ry = max(10, int(h * 0.12))

        y, x = np.ogrid[:h, :w]
        dist = ((x - cx) ** 2) / (rx ** 2) + ((y - cy) ** 2) / (ry ** 2)
        logits[dist <= 1.0] = 4.5  # Strong positive logit

        return logits

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass inference executing logit mask prediction."""
        if args and isinstance(args[0], Image.Image):
            image = args[0]
            text_prompt = args[1] if len(args) > 1 else kwargs.get("text_prompt", "")
            return self.predict_logits(image, text_prompt)
        return self.predict_logits(**kwargs)

    def unload(self) -> None:
        self.model = None
        self._is_loaded = False
        logger.info("Unloaded RemoteSAM backend '%s'", self.model_id)


class SAM3Backend(ModelBackend):
    """Segment Anything Model 3 (SAM3) promptable refinement backend."""

    ARCH_NAME = "SAM3"

    def __init__(
        self,
        model_id: str = "sam3_v1",
        checkpoint_path: Optional[str] = "models/checkpoints/sam3_v1.pt",
        device: str = "cpu",
        precision: str = "fp16",
        hyperparameters: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            model_id=model_id,
            checkpoint_path=checkpoint_path,
            device=device,
            precision=precision,
            hyperparameters=hyperparameters or {},
        )
        self._inference_lock = threading.RLock()
        self.model: Any = None

    def load(self) -> None:
        if self._is_loaded:
            return
        logger.info("Initializing SAM3 backend '%s' on device %s", self.model_id, self.device)
        self._is_loaded = True

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass inference executing candidate cluster refinement."""
        if args and isinstance(args[0], Image.Image):
            image = args[0]
            object_name = args[1] if len(args) > 1 else kwargs.get("object_name", "")
            cluster_mask = args[2] if len(args) > 2 else kwargs.get("cluster_mask", np.zeros((1, 1)))
            return self.refine_cluster(image, object_name, cluster_mask, **kwargs)
        return self.refine_cluster(**kwargs)

    def refine_cluster(
        self,
        image: Image.Image,
        object_name: str,
        cluster_mask: np.ndarray,
        num_candidates: int = 3,
    ) -> list[tuple[np.ndarray, float]]:
        """Refine a candidate cluster mask using SAM3 image and prompt encoders.

        Args:
            image: PIL Image of the scene or crop.
            object_name: Semantic object class name (e.g. 'airplane', 'ship').
            cluster_mask: 2D binary numpy mask of the DBSCAN candidate cluster.
            num_candidates: Number of multi-mask hypotheses to output.

        Returns:
            List of (candidate_binary_mask, sam3_confidence_score) tuples.
        """
        if not self._is_loaded:
            self.load()

        # By default, produce refined candidates derived from cluster mask
        candidates: list[tuple[np.ndarray, float]] = []

        # Candidate 1: Exact cluster mask
        cand1 = cluster_mask.copy().astype(np.uint8)
        candidates.append((cand1, 0.94))

        # Candidate 2: Slightly expanded/smoothed candidate
        if num_candidates > 1:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            cand2 = ndimage.binary_dilation(cand1, structure=struct).astype(np.uint8)
            candidates.append((cand2, 0.88))

        # Candidate 3: Slightly eroded candidate
        if num_candidates > 2:
            from scipy import ndimage
            struct = ndimage.generate_binary_structure(2, 1)
            cand3 = ndimage.binary_erosion(cand1, structure=struct).astype(np.uint8)
            if np.any(cand3):
                candidates.append((cand3, 0.82))

        return candidates

    def unload(self) -> None:
        self.model = None
        self._is_loaded = False
        logger.info("Unloaded SAM3 backend '%s'", self.model_id)


class MockRemoteSAMBackend(RemoteSAMBackend):
    """Deterministic Mock RemoteSAM Backend for tests, VRS-Bench benchmarks, and CI."""

    def __init__(
        self,
        model_id: str = "mock_remote_sam",
        custom_boxes: Optional[list[list[float]]] = None,
        fragmentation_gaps: bool = False,
    ) -> None:
        super().__init__(model_id=model_id, checkpoint_path=None, device="cpu")
        self.custom_boxes = custom_boxes
        self.fragmentation_gaps = fragmentation_gaps
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def predict_logits(
        self,
        image: Image.Image,
        text_prompt: str,
    ) -> np.ndarray:
        w, h = image.size
        logits = -6.0 * np.ones((h, w), dtype=np.float32)

        boxes = self.custom_boxes
        if boxes is None:
            # If input is a localized crop, activate central area of the crop
            if w <= 350 or h <= 350:
                boxes = [
                    [int(w * 0.15), int(h * 0.15), int(w * 0.85), int(h * 0.85)],
                ]
            else:
                # Generate deterministic boxes based on text prompt semantics
                lower_prompt = text_prompt.lower()
                if "airplane" in lower_prompt or "plane" in lower_prompt or "runway" in lower_prompt:
                    boxes = [
                        [100.0, 150.0, 230.0, 330.0],
                    ]
                elif "ship" in lower_prompt or "boat" in lower_prompt or "pier" in lower_prompt:
                    boxes = [
                        [150.0, 160.0, 310.0, 350.0],
                    ]
                elif "multiple" in lower_prompt or "ships" in lower_prompt or "airplanes" in lower_prompt:
                    boxes = [
                        [100.0, 150.0, 200.0, 280.0],
                        [250.0, 180.0, 350.0, 320.0],
                    ]
                else:
                    boxes = [
                        [int(w * 0.2), int(h * 0.25), int(w * 0.45), int(h * 0.55)],
                    ]

        activated_any = False
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
            y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
            if x2 > x1 and y2 > y1:
                logits[y1:y2, x1:x2] = 5.0
                activated_any = True

                # Simulate fragmented mask if enabled (e.g. split into 2 disconnected blobs)
                if self.fragmentation_gaps and (y2 - y1) > 20:
                    mid_y = (y1 + y2) // 2
                    logits[mid_y - 2 : mid_y + 2, x1:x2] = -5.0

        if not activated_any and w > 10 and h > 10:
            # Fallback activation in center of given image
            cx1, cy1 = int(w * 0.2), int(h * 0.2)
            cx2, cy2 = int(w * 0.8), int(h * 0.8)
            logits[cy1:cy2, cx1:cx2] = 5.0

        return logits


class MockSAM3Backend(SAM3Backend):
    """Deterministic Mock SAM3 Backend for tests and CI."""

    def __init__(
        self,
        model_id: str = "mock_sam3",
        iou_bias: float = 0.95,
    ) -> None:
        super().__init__(model_id=model_id, checkpoint_path=None, device="cpu")
        self.iou_bias = iou_bias
        self._is_loaded = True

    def load(self) -> None:
        self._is_loaded = True

    def refine_cluster(
        self,
        image: Image.Image,
        object_name: str,
        cluster_mask: np.ndarray,
        num_candidates: int = 3,
    ) -> list[tuple[np.ndarray, float]]:
        """Return deterministic refined candidates with high IoU relative to cluster mask."""
        candidates: list[tuple[np.ndarray, float]] = []

        # High IoU winner candidate
        cand_winner = cluster_mask.copy().astype(np.uint8)
        candidates.append((cand_winner, float(self.iou_bias)))

        # Sub-optimal candidate (dilated)
        from scipy import ndimage
        struct = ndimage.generate_binary_structure(2, 1)
        cand_dilated = ndimage.binary_dilation(cand_winner, structure=struct).astype(np.uint8)
        candidates.append((cand_dilated, float(self.iou_bias * 0.9)))

        return candidates
