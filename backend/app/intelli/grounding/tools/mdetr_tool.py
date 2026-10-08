import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import file_to_base64_png_jpeg, draw_boxes, ground_run


@register_tool('object_detection')
class MDETR(BaseTool):
    description = (
        "This tool perform object detection on remote sensing or satellite imagery"
        "It returns an image with bounding boxes drawn around detected objects."
        '''Following is the list of object classes that can be detected by give tool:
                1. Plane
                2. Ship
                3. Storage Tank
                4. Baseball Diamond
                5. Tennis Court
                6. Basketball Court
                7. Ground Track Field
                8. Harbor
                9. Bridge
                10. Large Vehicle
                11. Small Vehicle
                12. Helicopter
                13. Roundabout
                14. Soccer Ball Field
                15. Swimming Pool
                16. Container Crane

        ##IMPORTANT: Always prefer to use this tool if object is from the above list##
        ##IMPORTANT: Use it only if query can be decomposed to detection of exactly above mentioned objects##

        '''
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
        'description': 'A list of object classes to detect (e.g., ["Plane", "Harbor", "Bridge"]).',
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

        try:
            result = ground_run("object_detector", {
                "image_base64": file_to_base64_png_jpeg(image_path),
                "target_classes": object_class_list,
            })
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"Object Detector Currently Unavailable: {e}")]

        detections = [
            (d["class_name"], [d["bbox_px"][k] for k in ("x_min", "y_min", "x_max", "y_max")], d["confidence"]["score"])
            for d in result.get("detections") or []
        ]
        objects = [{"class": c, "horizontal_bbox": [round(v) for v in b], "score": round(s, 3)} for c, b, s in detections]
        output_path = draw_boxes(image_path, [(c, b) for c, b, _ in detections])
        return [
            ContentItem(text=f"Detected {len(objects)} objects: {json.dumps(objects)}"),
            ContentItem(image=output_path),
        ]
