import os
import json
import requests
from PIL import Image
from qwen_agent.tools.base import BaseTool, register_tool
from qwen_agent.llm.schema import ContentItem
from backend.app.intelli.common import TERRASCOPE_URL, file_to_base64, base64_to_image, save_image


@register_tool('bi_temporal_analyser')
class TERRASCOPE(BaseTool):
    description = (
    "This tool is capable of performing bi-temporal analysis on remote sensing or satellite imagery at the pixel level. "
    "It can compare two images from different time periods to identify, localize, and reason about changes in specific regions or objects. "
    "The query offered to the tool must explicitly require bi-temporal or pixel-level analysis of the provided images."
)

    parameters = [{
        'name': 'image_path_1',
        'type': 'string',
        'description': 'The local file path to the input image 1.',
        'required': True
    },
        {
        'name': 'image_path_2',
        'type': 'string',
        'description': 'The local file path to the input image 2.',
        'required': True
    },
        {
        'name': 'query',
        'type': 'string',
        'description': 'The query for the bi-temporal image analysis.',
        'required': True
    }
    ]

    def call(self, params: str | dict, **kwargs) -> list[ContentItem]:
        if isinstance(params, str):
            params = json.loads(params)

        image_path_1 = params.get('image_path_1')
        image_path_2 = params.get('image_path_2')
        query = params.get('query')

        for path in (image_path_1, image_path_2):
            if not path or not os.path.exists(path):
                return [ContentItem(text=f"Error: Image file not found at {path}")]

        # TerraScope needs equally sized images for pixel-level masks
        with Image.open(image_path_1) as img1, Image.open(image_path_2) as img2:
            size1, size2 = img1.size, img2.size
            if size1 != size2:
                if abs(size1[0] / size1[1] - size2[0] / size2[1]) > 0.01 * (size1[0] / size1[1]):
                    return [ContentItem(text=(
                        f"Error: the two images have different sizes ({size1} vs {size2}) and aspect ratios, "
                        "so a pixel-level comparison is not possible. Use the earthdial tool for this pair instead."
                    ))]
                image_path_2 = save_image(img2.convert("RGB").resize(size1, Image.Resampling.BICUBIC))

        try:
            response = requests.post(
                f"{TERRASCOPE_URL}/run",
                json={
                    "image_base64": file_to_base64(image_path_1),
                    "compare_image_base64": file_to_base64(image_path_2),
                    "prompt": query,
                },
                timeout=300
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            detail = e.response.text if e.response is not None else ""
            return [ContentItem(text=f"Bi-temporal Analyser Currently Unavailable: {e} {detail}")]

        result = response.json()
        items = [ContentItem(text=f"Analysis: {result.get('answer')}\n")]
        if size1 != size2:
            items.append(ContentItem(text=f"Note: image 2 ({size2}) was resized to image 1's size ({size1}) for the comparison."))
        if result.get("overlay_base64"):
            items.append(ContentItem(image=save_image(base64_to_image(result["overlay_base64"]))))
            items.append(ContentItem(text=f"Coverage Percent: {result.get('coverage_percent')}"))
        return items
