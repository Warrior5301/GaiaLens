import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import CONTEXT_URL, file_to_base64, base64_to_image, save_image


@register_tool('image_context')
class ImageContextTool(BaseTool):
    description = (
        "Use this tool when you cannot tell what an image shows or where it is. "
        "It looks the image up and returns four images of the same place at different zoom levels "
        "(from wide area to close-up) together with textual context about the place. "
        "Take the returned images and context into account when answering. "
        "Do not use it when the image content is already clear."
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The local file path to the image you could not identify.',
        'required': True
    }]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            response = requests.post(
                f"{CONTEXT_URL}/run",
                json={"image_base64": file_to_base64(image_path)},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            detail = getattr(getattr(e, 'response', None), 'text', '')
            return [ContentItem(text=f"Image Context Service Currently Unavailable: {e} {detail}")]

        # Images are base64 strings, or objects like {"image_base64": ..., "zoom_level": ...}
        items = []
        for index, entry in enumerate(result.get('images') or [], start=1):
            data = entry if isinstance(entry, str) else entry.get('image_base64') or entry.get('image')
            if not data:
                continue
            label = entry.get('zoom_level') or entry.get('label') if isinstance(entry, dict) else None
            items.append(ContentItem(text=f"Zoom view {index}" + (f" ({label})" if label else "") + ":"))
            items.append(ContentItem(image=save_image(base64_to_image(data))))

        if not items:
            return [ContentItem(text="The image context service returned no images for this image.")]

        context = result.get('context')
        context = context if isinstance(context, str) else json.dumps(context, ensure_ascii=False)
        return [ContentItem(text=f"Context for the image: {context}")] + items
