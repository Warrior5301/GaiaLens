import os
import json
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.orchestrator.tool_utils.init_image_analyze import analyze_image


@register_tool('image_analyzer')
class ImageAnalyzerTool(BaseTool):
    description = (
        "Analyzes and processes an input image using computer vision algorithms."
        "It extracts metadata, dimensions, aspect ratio, color bands, and other relevant information."
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The absolute file path to the image file to be analyzed.',
        'required': True
    }]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            with Image.open(image_path) as img:
                result = analyze_image(img)
            return [ContentItem(text=f"Image analysis: {json.dumps(result, default=str)}")]
        except Exception as e:
            return [ContentItem(text=f"An error occurred during image analysis: {str(e)}")]
