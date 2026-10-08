import base64
import json
import re
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from PIL import Image

from backend.app.intelli.common import (
    TMP_DIR,
    build_messages,
    cleanup_tmp,
    current_model,
    load_lora,
    run_messages,
)
from backend.app.intelli.grounding.ground import create_agent
from backend.app.intelli.grounding.tool_utils.draw_boxes import draw_bboxes

router = APIRouter()
_agents = {}

# Which coordinate system each key of the agent's final JSON uses:
#   horizontal_bbox - copied from a tool result: pixels of the original image
#   bbox_2d / bbox_1000 / bbox / box - written by the model itself: Qwen3-VL's native relative 0-1000
BOX_KEYS = {"horizontal_bbox": "pixel", "bbox_2d": "qwen", "bbox_1000": "qwen", "bbox": "qwen", "box": "qwen"}

ANSWER_FORMAT = (
    "\n\nImage size: {width}x{height} pixels.\n"
    'Give the final answer as JSON: {{"objects": [{{"class": "...", "horizontal_bbox": [x_min, y_min, x_max, y_max]}}]}}.\n'
    'For boxes taken from tool results use the key "horizontal_bbox" with the tool\'s pixel coordinates unchanged. '
    'For any box you estimate yourself from the image without a tool, use the key "bbox_2d" with your native '
    "Qwen3-VL relative coordinates (0-1000, [x_min, y_min, x_max, y_max]); do not convert those to pixels."
)


def _get_agent(model: str):
    """One grounding agent per served model (base model or LoRA adapter name)."""
    if model not in _agents:
        token = current_model.set(model)
        try:
            _agents[model] = create_agent()
        finally:
            current_model.reset(token)
    return _agents[model]


def _json_candidates(text: str):
    """Yield every JSON value found in the text: fenced blocks first, then any object/array in the prose."""
    chunks = re.findall(r"```(?:json)?\s*(.*?)```", text, re.S) + [text]
    decoder = json.JSONDecoder()
    for chunk in chunks:
        for match in re.finditer(r"[\[{]", chunk):
            try:
                yield decoder.raw_decode(chunk[match.start():])[0]
            except ValueError:
                continue


def parse_objects(text: str) -> list[dict]:
    """Extract [{"label", "bbox", "coord_system"}] from the agent's final answer."""
    for data in _json_candidates(text):
        if isinstance(data, dict):
            items = data.get("objects") or data.get("detections") or [data]
        else:
            items = data
        if not isinstance(items, list):
            continue

        objects = []
        for item in items:
            if not isinstance(item, dict):
                continue
            key = next((k for k in BOX_KEYS if isinstance(item.get(k), list) and len(item[k]) == 4), None)
            if key:
                label = item.get("class") or item.get("label") or item.get("name") or ""
                objects.append({"label": label, "bbox": item[key], "coord_system": BOX_KEYS[key]})
        if objects:
            return objects
    return []


@router.post("/ground")
def ground(query: str = Form(...), image: UploadFile = File(...)):
    """Run the grounding agent directly (no orchestrator) and return the image with its boxes drawn."""
    if not query.strip():
        raise HTTPException(422, "query must not be empty")

    cleanup_tmp()
    data = image.file.read()
    try:
        with Image.open(BytesIO(data)) as opened:
            width, height = opened.size
            ext = (opened.format or "png").lower()
    except Exception:
        raise HTTPException(422, "Uploaded file is not a supported image")
    image_path = TMP_DIR / f"{uuid.uuid4().hex}.{ext}"
    image_path.write_bytes(data)

    model = load_lora("Grounding")
    token = current_model.set(model)
    try:
        agent = _get_agent(model)
        prompt = query.strip() + ANSWER_FORMAT.format(width=width, height=height)
        answer = run_messages(agent, build_messages(prompt, str(image_path)), list_images=False)
    except Exception as e:
        raise HTTPException(500, f"Grounding agent failed: {e}")
    finally:
        current_model.reset(token)

    objects = parse_objects(answer)
    if not objects:
        return {"response": answer, "objects": [], "image": None, "model": model}

    output_path, drawn = draw_bboxes(str(image_path), objects)
    name = Path(output_path).name
    return {
        "response": answer,
        "objects": drawn,
        "image": {
            "name": name,
            "url": f"/files/{name}",
            "mime_type": "image/png",
            "base64": base64.b64encode(Path(output_path).read_bytes()).decode("ascii"),
        },
        "model": model,
    }
