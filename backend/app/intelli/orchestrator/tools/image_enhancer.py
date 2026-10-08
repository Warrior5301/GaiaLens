import os
import json
import requests
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import ENHANCER_URL, file_to_base64, base64_to_image, save_image


@register_tool('image_enhancer')
class SlimDiffSREnhancer(BaseTool):
    description = (
        "Enhances and upscales low-resolution remote sensing or satellite imagery "
        "using the SlimDiffSR diffusion model. Call this tool "
        "to increase the resolution, sharpen, or deblur satellite photos, aerial "
        "maps, or remote sensing rasters by a factor of 4x. "
        "Input must be file path to the image"
    )

    parameters = [{
        'name': 'image_path',
        'type': 'string',
        'description': 'The file path to the input low-resolution image.',
        'required': True
    }, ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path = params.get('image_path')

        if not image_path or not os.path.exists(image_path):
            return [ContentItem(text=f"Error: Image file not found at {image_path}")]

        try:
            requests.post(f"{ENHANCER_URL}/load", timeout=300).raise_for_status()
            response = requests.post(
                f"{ENHANCER_URL}/run",
                json={"image_base64": file_to_base64(image_path)},
                timeout=300
            )
            response.raise_for_status()
            output_path = save_image(base64_to_image(response.json()["image_base64"]))
            return [
                ContentItem(text=f"Successfully enhanced image, saved to {output_path}"),
                ContentItem(image=output_path)
            ]

        except requests.exceptions.RequestException as e:
            return [ContentItem(text=f"Image Enhancer Service Currently Unavailable: {e}")]
        finally:
            try:
                requests.post(f"{ENHANCER_URL}/unload", timeout=10)
            except Exception:
                # Silently catch cleanup errors to avoid masking main errors
                pass
