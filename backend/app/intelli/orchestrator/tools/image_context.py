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
        "It estimates the image location, retrieves a nearby satellite image, and returns brief "
        "web and location context. Take the returned image and context into account when answering. "
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
                f"{CONTEXT_URL}/location-explorer",
                json={"image_base64": file_to_base64(image_path)},
                timeout=300
            )
            response.raise_for_status()
            result = response.json()
        except (requests.exceptions.RequestException, ValueError) as e:
            detail = getattr(getattr(e, 'response', None), 'text', '')
            return [ContentItem(text=f"Image Context Service Currently Unavailable: {e} {detail}")]

        # The location explorer returns nearby satellite images as Base64 objects.
        items = []
        for index, entry in enumerate(result.get('satellite_images') or [], start=1):
            data = entry.get('image_base64') if isinstance(entry, dict) else None
            if not data:
                continue
            items.append(ContentItem(text=f"Nearby satellite image {index}:"))
            items.append(ContentItem(image=save_image(base64_to_image(data))))

        if not items:
            return [ContentItem(text="The location context service returned no nearby satellite image.")]

        context = {
            "coordinates": result.get("coordinates"),
            "location_hint": result.get("location_hint"),
            "web_context": result.get("web_context"),
            "web_context_status": result.get("web_context_status"),
        }
        return [ContentItem(text=f"Location context for the image: {json.dumps(context, ensure_ascii=False)}")] + items
