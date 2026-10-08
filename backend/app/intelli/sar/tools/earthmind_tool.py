import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import EARTHMIND_URL, file_to_base64


@register_tool('multispectral_image_analyser')
class EARTH_SAR(BaseTool):
    description = (
        "This tool is capable of performing analysis on multispectral remote sensing or satellite imagery."
        "The query offered to the tool must explicit analysis of the image."
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
        'description': 'The query for the multispectral image analysis.',
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
                f"{EARTHMIND_URL}/run",
                json={"query_type": "normal", "prompt": query, "image_base64": file_to_base64(image_path)},
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"Multispectral Image Analyser Currently Unavailable: {e}")]

        return [ContentItem(text=f"Generated prediction: {response.json().get('prediction')}\n")]
