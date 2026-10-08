import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import EARTHDIAL_URL, file_to_base64


@register_tool('earthdial')
class EarthdialTool(BaseTool):
    description = (
        "Earthdial tool for general reasoning and temporal comparison over remote-sensing images. "
        "It analyzes two or more images and identifies semantic and structural changes based on a query."
    )

    parameters = [{
        'name': 'image_paths',
        'type': 'array',
        'items': {'type': 'string'},
        'description': 'A list of absolute file paths to the remote-sensing images to be compared (requires at least 2 images).',
        'required': True
    }, {
        'name': 'query',
        'type': 'string',
        'description': 'The query or instruction for reasoning and comparison over the images.',
        'required': True
    }]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_paths = params.get('image_paths')
        query = params.get('query')

        if not isinstance(image_paths, list) or len(image_paths) < 2:
            return [ContentItem(text="Error: 'image_paths' must be a list containing at least two image paths.")]

        for path in image_paths:
            if not os.path.exists(path):
                return [ContentItem(text=f"Error: Image file not found at {path}")]

        # The service may run in a container, so send image data rather than host paths
        payload = {
            "images": [f"data:image/png;base64,{file_to_base64(p)}" for p in image_paths],
            "query": query,
        }

        try:
            response = requests.post(f"{EARTHDIAL_URL}/compare", json=payload, timeout=600)
            response.raise_for_status()
        except requests.RequestException as e:
            return [ContentItem(text=f"An HTTP error occurred while calling Earthdial service: {str(e)}")]

        result = response.json()
        return [ContentItem(text=f"EarthDial Temporal Comparison Results:\n{result.get('response', json.dumps(result))}")]
