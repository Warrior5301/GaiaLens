"""Shared config and helpers for all GaiaLens agents and tools."""

import base64
import logging
import os
import time
import uuid
from contextvars import ContextVar
from io import BytesIO
from pathlib import Path

import requests
from PIL import Image, ImageDraw
from qwen_agent.utils.output_beautify import multimodal_typewriter_print

# backend/app/tmp, independent of the working directory
TMP_DIR = Path(__file__).resolve().parents[1] / "tmp"
TMP_DIR.mkdir(parents=True, exist_ok=True)


def _url(name: str, port: int) -> str:
    return os.getenv(f"{name}_URL", f"http://localhost:{port}")


# One port per service; override any of them with <NAME>_URL.
QWEN_URL = os.getenv("QWEN_URL", "http://localhost:8000/v1")
GROUND_URL = os.getenv(
    "GROUND_URL",
    "https://rajputvansh4391--sac-geoai-perception-api.modal.run",
)  # SAC GeoAI perception service
SAM3_URL = _url("SAM3", 8002)              # services/poly_service
OCR_URL = os.getenv(
    "OCR_URL",
    "https://rajputvansh4391--ppocrv5-api-fastapi-app.modal.run",
)  # PP-OCRv5 service
MOONDREAM_URL = _url("MOONDREAM", 8004)    # services/moondream
EARTHMIND_URL = _url("EARTHMIND", 8005)    # services/earthmind
TERRASCOPE_URL = os.getenv(
    "TERRASCOPE_URL",
    "https://rajputvansh4391--terrascope-bf16-test-fastapi-app.modal.run",
)  # TerraScope temporal comparison
EARTHDIAL_URL = os.getenv(
    "EARTHDIAL_URL",
    "https://rajputvansh4391--earthdial-temporal-service-earthdialmod-24ccc1.modal.run",
)  # EarthDial temporal comparison
ENHANCER_URL = _url("ENHANCER", 8008)      # services/image_enhancer
GSD_URL = _url("GSD", 8009)                # services/gsd_service
CONTEXT_URL = os.getenv(
    "CONTEXT_URL",
    "https://rajputvansh4391--location-context-fastapi-fastapi-app.modal.run",
)  # image location and nearby satellite context
YOLOVR_URL = os.getenv(
    "YOLOVR_URL",
    os.getenv("YOLOVR_API_URL", "https://rajputvansh4391--yolovr-inference-web-app.modal.run"),
).rstrip("/")  # services/yolo_serve/yolovr; accepts either a base URL or a /predict URL


BASE_MODEL = os.getenv("QWEN_MODEL", 'Qwen/Qwen3-VL-8B-Instruct')

# Model name vLLM serves for the current request: the base model, or a hot-loaded LoRA adapter's name
current_model: ContextVar[str] = ContextVar("current_model", default=BASE_MODEL)


# Paths of images produced by tools during the current request (None when nobody is collecting them)
output_images: ContextVar[list | None] = ContextVar("output_images", default=None)


def record_output_images(paths: list[str]) -> None:
    collected = output_images.get()
    if collected is not None:
        collected.extend(p for p in paths if p not in collected)


def cleanup_tmp(max_age_hours: float | None = None) -> None:
    """Delete files in TMP_DIR older than max_age_hours (env TMP_MAX_AGE_HOURS, default 24)."""
    max_age = float(os.getenv("TMP_MAX_AGE_HOURS", "24")) if max_age_hours is None else max_age_hours
    cutoff = time.time() - max_age * 3600
    for path in TMP_DIR.iterdir():
        try:
            if path.is_file() and path.stat().st_mtime < cutoff:
                path.unlink()
        except OSError:
            pass


def llm_cfg() -> dict:
    return {
        'model_type': 'qwenvl_oai',
        'model': current_model.get(),
        'model_server': QWEN_URL,
        'api_key': 'EMPTY',
        'generate_cfg': {
            'top_p': 0.8,
            'top_k': 20,
            'temperature': 0.7,
            'repetition_penalty': 1.0,
            'presence_penalty': 1.5,
        },
    }


def file_to_base64(path: str) -> str:
    """Base64 of the file's original bytes (keeps format, bands and bit depth)."""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def file_to_base64_png_jpeg(path: str) -> str:
    """Like file_to_base64, but re-encodes to PNG unless the file is already PNG or JPEG."""
    with Image.open(path) as image:
        if image.format in ("PNG", "JPEG"):
            return file_to_base64(path)
        buffer = BytesIO()
        image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def base64_to_image(data: str) -> Image.Image:
    if data.startswith("data:"):
        data = data.split(",", 1)[1]
    return Image.open(BytesIO(base64.b64decode(data)))


def save_image(image: Image.Image) -> str:
    """Save to TMP_DIR as PNG and return the absolute path."""
    path = TMP_DIR / f"{uuid.uuid4().hex}.png"
    image.save(path, format="PNG")
    return str(path)


def draw_boxes(image_path: str, boxes: list[tuple[str, list[float]]]) -> str:
    """Draw (label, [x_min, y_min, x_max, y_max]) boxes and return the saved path."""
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for label, box in boxes:
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 2, box[1] + 2), label, fill="red")
    return save_image(image)


def ground_run(tool: str, payload: dict, timeout: int = 300) -> dict:
    """Run a ground_service tool, loading its model first if needed."""
    response = requests.post(f"{GROUND_URL}/tools/{tool}/run", json=payload, timeout=timeout)
    if response.status_code == 409:
        requests.post(f"{GROUND_URL}/tools/{tool}/load", json={}, timeout=timeout).raise_for_status()
        response = requests.post(f"{GROUND_URL}/tools/{tool}/run", json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


logger = logging.getLogger("gaialens")

# Local directory or HF hub ID of each LoRA adapter, set via environment variables
LORA_PATH_ENV = {
    "Grounding": "LORA_GROUNDING_PATH",
    "VQA": "LORA_VQA_PATH",
    "Captioning": "LORA_CAPTIONING_PATH",
    "SAR/IR": "LORA_SAR_IR_PATH",
}


def load_lora(lora: str) -> str:
    """Hot-load the given LoRA into vLLM; returns the model name to query (the base model on fallback)."""
    lora_path = os.getenv(LORA_PATH_ENV.get(lora, ""))
    if not lora_path:
        logger.warning("No path configured for LoRA %r (set %s); using the base model", lora, LORA_PATH_ENV.get(lora))
        return BASE_MODEL

    lora_name = "qwen3-vl-" + lora.lower().replace("/", "-")
    try:
        response = requests.post(
            f"{QWEN_URL}/load_lora_adapter",
            json={"lora_name": lora_name, "lora_path": lora_path},
            timeout=300,
        )
    except requests.RequestException as e:
        logger.warning("Could not reach vLLM to load LoRA %r: %s; using the base model", lora_name, e)
        return BASE_MODEL

    # vLLM answers 400 "already been loaded" when the adapter is registered from an earlier request
    if response.ok or "already" in response.text.lower():
        return lora_name
    logger.warning("Loading LoRA %r failed (%s): %s; using the base model", lora_name, response.status_code, response.text)
    return BASE_MODEL


def build_messages(prompt: str, image_paths=None) -> list[dict]:
    """User message with the images plus their absolute paths, so the model can pass them to tools."""
    if isinstance(image_paths, (str, Path)):
        image_paths = [image_paths]
    paths = [str(Path(p).resolve()) for p in image_paths or []]
    content = [{'image': p} for p in paths]
    if paths:
        listing = "\n".join(f"Image {i + 1} path: {p}" for i, p in enumerate(paths))
        prompt = f"{listing}\n\n{prompt}"
    content.append({'text': prompt})
    print(f"\nUser: {prompt}\n")
    return [{'role': 'user', 'content': content}]


def run_messages(agent, messages: list[dict], list_images: bool = True) -> str:
    """
    Run the agent and return its final answer. Images produced by tools are always recorded
    for the current request (see output_images); with list_images they are also appended to
    the text as paths, which lets a parent agent pass them on to other tools.
    """
    text, responses = '', []
    for responses in agent.run(messages):
        text = multimodal_typewriter_print(responses, text)

    answer, images = '', []
    # Messages and content items may be dicts or qwen_agent models; both support .get()
    for msg in responses:
        content = msg.get('content')
        if msg.get('role') == 'assistant' and isinstance(content, str) and content:
            answer = content
        elif msg.get('role') == 'function' and isinstance(content, list):
            images += [item.get('image') for item in content if hasattr(item, 'get') and item.get('image')]
    images = list(dict.fromkeys(images))
    record_output_images(images)
    if images and list_images:
        answer += "\n\nOutput images:\n" + "\n".join(images)
    return answer
