import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import SAM3_URL, file_to_base64, base64_to_image, save_image


@register_tool('segmentation')
class Segmentation(BaseTool):
    description = (
        "This tool performs segmentation task by drawing polygon over queried object in remote sensing or satellite imagery."
        "It returns an image drawen with polygon around detected object."
        "It also returns the coordinates of the polygon in a list of tuples format."
        "##IMPORTANT: Always prefer this tool if the queried object class can not be bounded by rectangle and is of irregular shape##"
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The local file path to the input image.',
        'required': True
    },
        {
        'name': 'object_class',
        'type': 'string',
        'description': 'The class of the object to segment (e.g., "river").',
        'required': True
    }
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')
        object_class = params.get('object_class')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            response = requests.post(
                f"{SAM3_URL}/run",
                json={"image": file_to_base64(image_path), "prompt": object_class},
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"Segmentation Tool Currently Unavailable: {e}")]

        result = response.json()
        output_path = save_image(base64_to_image(result["image"]))
        objects = [
            {"score": round(o["score"], 3), "box": [round(v) for v in o["box"]], "polygons": o["polygons"]}
            for o in result.get("objects", [])
        ]
        return [
            ContentItem(text=f"Segmented {len(objects)} '{object_class}' objects. Polygon coordinates: {json.dumps(objects)}"),
            ContentItem(image=output_path),
        ]
