import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import OCR_URL, file_to_base64


@register_tool('ocr_tool')
class OCRTool(BaseTool):
    description = (
        "This tool performs OCR on remote sensing or satellite imagery"
        "It returns the text detected in the image."
        "First you can zoom into the image using image_zoom_in_tool tool and then use this tool to extract text from the image."
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The local file path to the input image.',
        'required': True
    }
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            response = requests.post(
                f"{OCR_URL}/run",
                json={"image": file_to_base64(image_path)},
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"OCR Tool Currently Unavailable: {e}")]

        result = response.json()
        return [
            ContentItem(text=f"Detected text: {result.get('text')}\n"),
            ContentItem(text=f"Detected lines: {json.dumps(result.get('lines'))}"),
        ]
