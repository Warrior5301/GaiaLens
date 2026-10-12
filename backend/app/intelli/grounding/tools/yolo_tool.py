"""Qwen tool adapter for the YOLO-World Modal /predict HTTP endpoint.

Unlike ground_service/object_detector, this endpoint returns only an annotated
image. The agent must not claim machine-readable coordinates from this tool.
"""

import json
import os

import requests
from qwen_agent.llm.schema import ContentItem
from qwen_agent.tools.base import BaseTool, register_tool

from backend.app.intelli.common import YOLOVR_URL, base64_to_image, file_to_base64_png_jpeg, save_image


YOLO_CLASSES = [
    "airplane", "airport", "baseball-diamond", "basketball-court", "bridge", "chimney", "container-crane", "dam",
    "expressway-service-area", "expressway-toll-station", "golf-field", "ground-track-field", "harbor",
    "helicopter", "helipad", "overpass", "roundabout", "ship", "soccer-ball-field", "stadium", "storage-tank",
    "swimming-pool", "tennis-court", "train-station", "vehicle", "windmill",
]


def _canonical(name: str) -> str:
    return name.strip().lower().replace("_", "-").replace(" ", "-")


@register_tool("yolo_object_detection")
class YOLODetection(BaseTool):
    description = (
        "Detects supported remote-sensing object classes with a separately deployed YOLO-World model. "
        "Returns one annotated image per requested class, but the API does not return bounding-box coordinates. "
        "Use object_detection instead when numeric boxes are required and it supports the class. "
        "Supported classes: " + ", ".join(YOLO_CLASSES)
    )
    parameters = [
        {"name": "image_path", "type": "string", "description": "Local path to the image", "required": True},
        {"name": "object_class_list", "type": "array", "items": {"type": "string"},
         "description": "Classes to detect; one request is made per class", "required": True},
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)
        image_path = params.get("image_path")
        if not image_path or not os.path.isfile(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        requested = params.get("object_class_list") or []
        if isinstance(requested, str):
            requested = [requested]
        if not isinstance(requested, list) or not all(isinstance(item, str) for item in requested):
            return [ContentItem(text="Error: object_class_list must contain class names.")]
        names = list(dict.fromkeys(_canonical(item) for item in requested))
        classes = [name for name in names if name in YOLO_CLASSES]
        ignored = [name for name in names if name not in YOLO_CLASSES]
        if not classes:
            return [ContentItem(text=f"Unsupported YOLO classes: {ignored}. Use open_language_grounding for other objects.")]

        endpoint = YOLOVR_URL if YOLOVR_URL.endswith("/predict") else f"{YOLOVR_URL}/predict"
        try:
            encoded = file_to_base64_png_jpeg(image_path)
        except Exception as exc:
            return [ContentItem(text=f"Could not read image for YOLO: {exc}")]

        outputs = []
        for name in classes:
            try:
                response = requests.post(endpoint, json={"image_base64": encoded, "prompt": name}, timeout=300)
                response.raise_for_status()
                result = response.json()
                annotated = result.get("image_base64") or result.get("image")
                if not isinstance(annotated, str) or not annotated:
                    raise ValueError("YOLO returned no annotated image")
                output_path = save_image(base64_to_image(annotated))
            except (requests.RequestException, ValueError, OSError) as exc:
                outputs.append(ContentItem(text=f"YOLO detection failed for {name}: {exc}"))
                continue
            outputs.extend([
                ContentItem(text=f"YOLO annotated image for {name}; numeric boxes were not returned by the service."),
                ContentItem(image=output_path),
            ])
        if ignored:
            outputs.append(ContentItem(text=f"Unsupported classes skipped: {ignored}."))
        return outputs
