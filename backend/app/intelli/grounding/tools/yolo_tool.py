import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import file_to_base64_png_jpeg, draw_boxes, ground_run

# Route of the YOLO detector on the detection service: POST {GROUND_URL}/tools/<YOLO_TOOL>/run
# (same request/response format as object_detection)
YOLO_TOOL = "yolo_detector"

YOLO_CLASSES = [
    "airplane", "airport", "baseball-diamond", "basketball-court", "bridge", "chimney", "container-crane", "dam",
    "expressway-service-area", "expressway-toll-station", "golf-field", "ground-track-field", "harbor",
    "helicopter", "helipad", "overpass", "roundabout", "ship", "soccer-ball-field", "stadium", "storage-tank",
    "swimming-pool", "tennis-court", "train-station", "vehicle", "windmill",
]


def _canonical(name: str) -> str:
    """'Baseball Diamond' / 'baseball_diamond' -> 'baseball-diamond'"""
    return name.strip().lower().replace("_", "-").replace(" ", "-")


@register_tool('yolo_object_detection')
class YOLODetection(BaseTool):
    description = (
        "This tool performs object detection on remote sensing or satellite imagery using a YOLO detector. "
        "It returns an image with bounding boxes drawn around detected objects. "
        "Following is the list of object classes that can be detected by this tool:\n"
        + "\n".join(f"{i}. {c}" for i, c in enumerate(YOLO_CLASSES, 1))
        + "\n\n"
        "##IMPORTANT: Use this tool for objects from the above list, especially those that object_detection cannot detect "
        "(for example airport, dam, chimney, windmill, stadium, helipad, golf-field)##\n"
        "##IMPORTANT: Use it only if the query can be decomposed to detection of exactly the above mentioned objects##"
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The local file path to the input image.',
        'required': True
    },
        {
        'name': 'object_class_list',
        'type': 'array',
        'items': {'type': 'string'},
        'description': 'A list of object classes to detect, from the supported list (e.g., ["airport", "dam", "windmill"]).',
        'required': True
    }
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')
        object_class_list = params.get('object_class_list') or []
        if isinstance(object_class_list, str):
            object_class_list = [object_class_list]

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        # The detector only knows these class names (hyphenated, lower case)
        requested = [_canonical(c) for c in object_class_list]
        classes = list(dict.fromkeys(c for c in requested if c in YOLO_CLASSES))
        ignored = list(dict.fromkeys(c for c in requested if c not in YOLO_CLASSES))
        if not classes:
            return [ContentItem(text=(
                f"Error: none of the requested classes {ignored} are supported by this tool. "
                f"Supported classes: {YOLO_CLASSES}. Use open_language_grounding for other objects."
            ))]

        try:
            result = ground_run(YOLO_TOOL, {
                "image_base64": file_to_base64_png_jpeg(image_path),
                "target_classes": classes,
            })
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"YOLO Object Detector Currently Unavailable: {e}")]

        detections = [
            (d["class_name"], [d["bbox_px"][k] for k in ("x_min", "y_min", "x_max", "y_max")], d["confidence"]["score"])
            for d in result.get("detections") or []
        ]
        objects = [{"class": c, "horizontal_bbox": [round(v) for v in b], "score": round(s, 3)} for c, b, s in detections]
        output_path = draw_boxes(image_path, [(c, b) for c, b, _ in detections])
        note = f" Ignored unsupported classes: {ignored}." if ignored else ""
        return [
            ContentItem(text=f"Detected {len(objects)} objects: {json.dumps(objects)}.{note}"),
            ContentItem(image=output_path),
        ]
