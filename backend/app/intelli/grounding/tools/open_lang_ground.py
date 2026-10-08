import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import SAM3_URL, file_to_base64, draw_boxes


@register_tool('open_language_grounding')
class OpenGround(BaseTool):
    description = (
    "Perform open-vocabulary object grounding on remote sensing or satellite imagery. "
    "Unlike object_detection, this tool is not restricted to a predefined set of object classes. "
    "Use a natural-language query to describe the object you want to locate. "
    "Use this tool when the requested object is unavailable in object_detection, "
    "or when the object requires a specific or descriptive query. "
    "Write the query as a short noun phrase (e.g. 'white storage tank', 'airport runway'), "
    "one object type per call. "
    "The tool returns an image with bounding boxes around regions matching the query."
)

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The local file path to the input image.',
        'required': True
    },
        {
        'name': 'query',
        'type': 'string',
        'description': 'The natural-language query describing the object to locate.',
        'required': True
    }
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')
        query = params.get('query')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            response = requests.post(
                f"{SAM3_URL}/run",
                json={"image": file_to_base64(image_path), "prompt": query, "simplify": 0},
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"Open Language Grounding Currently Unavailable: {e}")]

        found = response.json().get("objects", [])
        objects = [{"class": query, "horizontal_bbox": [round(v) for v in o["box"]], "score": round(o["score"], 3)} for o in found]
        output_path = draw_boxes(image_path, [(query, o["box"]) for o in found])
        return [
            ContentItem(text=f"Grounded {len(objects)} objects: {json.dumps(objects)}"),
            ContentItem(image=output_path),
        ]
