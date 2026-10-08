import base64
from io import BytesIO
from PIL import Image


def image_to_base64(image: Image.Image, format: str = "PNG") -> str:
    """Convert a PIL Image to a base64-encoded string."""
    buffer = BytesIO()
    image.save(buffer, format=format)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(base64_string: str) -> Image.Image:
    """Convert a base64-encoded string back to a PIL Image."""
    image_data = base64.b64decode(base64_string)
    return Image.open(BytesIO(image_data))