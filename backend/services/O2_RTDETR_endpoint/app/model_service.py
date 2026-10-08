import gc
import logging
import threading

import cv2
import numpy as np
import torch
from PIL import Image

from app.config import CHECKPOINT_PATH, CONFIG_PATH, DEVICE
from app.schemas import Detection

logger = logging.getLogger("o2_rtdetr_api")


class O2RTDETRService:
    """Own one O2-RTDETR model and serialize inference/visualizer access."""

    def __init__(self) -> None:
        self.model = None
        self.visualizer = None
        self.device = DEVICE if DEVICE != "cuda:0" or torch.cuda.is_available() else "cpu"
        self._lock = threading.Lock()

    @property
    def ready(self) -> bool:
        return self.model is not None and self.visualizer is not None

    def load(self) -> None:
        if not CONFIG_PATH.is_file():
            raise FileNotFoundError(f"O2-RTDETR config not found: {CONFIG_PATH}")
        if not CHECKPOINT_PATH.is_file():
            raise FileNotFoundError(f"O2-RTDETR checkpoint not found: {CHECKPOINT_PATH}")

        logger.info("Loading O2-RTDETR from %s on %s", CHECKPOINT_PATH, self.device)
        from mmdet.apis import init_detector
        from mmrotate.registry import VISUALIZERS
        from mmrotate.utils import register_all_modules

        register_all_modules()
        model = init_detector(str(CONFIG_PATH), str(CHECKPOINT_PATH), device=self.device)
        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.dataset_meta = model.dataset_meta
        self.model = model
        self.visualizer = visualizer
        logger.info("O2-RTDETR ready on %s", self.device)

    def unload(self) -> None:
        with self._lock:
            self.visualizer = None
            self.model = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        logger.info("O2-RTDETR resources released")

    def predict(self, image: Image.Image, threshold: float):
        if not self.ready:
            raise RuntimeError("O2-RTDETR is not loaded")

        from mmdet.apis import inference_detector

        image_rgb = np.asarray(image.convert("RGB"))
        image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

        with self._lock, torch.inference_mode():
            result = inference_detector(self.model, image_bgr)
            instances = result.pred_instances
            boxes_tensor = instances.bboxes
            if hasattr(boxes_tensor, "tensor"):
                boxes_tensor = boxes_tensor.tensor
            boxes = boxes_tensor.detach().cpu().numpy()
            scores = instances.scores.detach().cpu().numpy()
            labels = instances.labels.detach().cpu().numpy()
            class_names = self.model.dataset_meta.get("classes", ())

            keep = scores >= threshold
            detections = []
            for box, score, label in zip(boxes[keep], scores[keep], labels[keep]):
                label_id = int(label)
                name = class_names[label_id] if label_id < len(class_names) else str(label_id)
                detections.append(
                    Detection(
                        label=str(name),
                        score=float(score),
                        box=[float(value) for value in box],
                    )
                )

            self.visualizer.add_datasample(
                "result",
                image_rgb,
                data_sample=result,
                draw_gt=False,
                show=False,
                wait_time=0,
                out_file=None,
                pred_score_thr=threshold,
            )
            rendered_rgb = self.visualizer.get_image()

        return rendered_rgb, detections


o2_rtdetr_service = O2RTDETRService()
