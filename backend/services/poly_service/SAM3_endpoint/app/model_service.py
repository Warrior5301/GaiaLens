# loads the model

import logging
import threading

import numpy as np
import torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

from app.config import settings

logger = logging.getLogger("sam3_api")


class Sam3Service:
    def __init__(self) -> None:
        self.model = None
        self.processor = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None

    def load(self) -> None:
        """Load model once at startup."""
        logger.info("Loading %s on %s ...", settings.MODEL_ID, self.device)
        processor = Sam3Processor.from_pretrained(settings.MODEL_ID)
        model = Sam3Model.from_pretrained(settings.MODEL_ID).to(self.device).eval()
        self.processor = processor
        self.model = model
        logger.info("SAM3 loaded")

    def unload(self) -> None:
        """Release model and processor resources."""
        with self._lock:
            self.model = None
            self.processor = None
            if self.device == "cuda":
                torch.cuda.empty_cache()

    def segment(self, image: Image.Image, prompt: str, threshold: float):
        """Returns (masks, boxes, scores) for all objects matching the prompt."""
        with self._lock, torch.no_grad():
            inputs = self.processor(
                images=image, text=prompt, return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            results = self.processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=0.5,
                target_sizes=inputs.get("original_sizes").tolist(),
            )[0]

        masks = [m.cpu().numpy() for m in results["masks"]]
        boxes = results["boxes"].cpu().numpy().tolist()
        scores = results["scores"].cpu().numpy().tolist()
        return masks, boxes, scores


sam3_service = Sam3Service()
