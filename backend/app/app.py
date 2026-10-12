"""HTTP gateway: refine a chat request, run agents, and publish generated images.

Both interactive chat experiences use this gateway.
Session state here belongs to this Python process and is not shared across workers.
"""

import base64
import json
import shutil
import uuid
from io import BytesIO
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from PIL import Image

from backend.app.intelli.common import TMP_DIR, cleanup_tmp, load_lora
from backend.app.intelli.grounding.api import router as grounding_router
from backend.app.query_refine import refine_query
from backend.app.start_conv import start_conversation_with_images

app = FastAPI(title="GaiaLens")
app.include_router(grounding_router)

# Chat history per session, kept in memory (lost on restart; fine for a prototype)
HISTORY: dict[str, list[dict]] = {}
HISTORY_WINDOW = 6   # latest messages sent to the query refiner
HISTORY_MAX = 20     # latest messages kept per session
MAX_SESSIONS = 200   # oldest session is dropped beyond this


def _preview_base64(data: bytes) -> str | None:
    """RGB PNG preview for Moondream; None if the image can't be shown as RGB (e.g. many-band rasters)."""
    try:
        with Image.open(BytesIO(data)) as image:
            buffer = BytesIO()
            image.convert("RGB").save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    except Exception:
        return None


def _publish_image(path: str) -> str | None:
    """Make a tool-produced image servable from TMP_DIR; returns its file name."""
    source = Path(path).resolve()
    if not source.is_file():
        return None
    if source.parent == TMP_DIR.resolve():
        return source.name
    target = TMP_DIR / f"{uuid.uuid4().hex}{source.suffix or '.png'}"
    shutil.copyfile(source, target)
    return target.name


@app.get("/files/{name}")
def get_file(name: str):
    """Serve an image produced during a chat (names are unguessable UUIDs)."""
    path = (TMP_DIR / name).resolve()
    if path.parent != TMP_DIR.resolve() or not path.is_file():
        raise HTTPException(404, "File not found")
    return FileResponse(path)


@app.post("/chat")
def chat(
    query: str = Form(...),
    history: str | None = Form(None, description='JSON list of {"role", "content"}; overrides the stored history'),
    session_id: str | None = Form(None, description="Identifies the conversation; omit on the first message and reuse the returned one"),
    images: list[UploadFile] = File(default=[]),
):
    try:
        history_list = json.loads(history) if history else None
    except json.JSONDecodeError:
        raise HTTPException(422, "history must be a JSON list")

    if history_list is not None:
        if not isinstance(history_list, list) or any(
            not isinstance(turn, dict)
            or turn.get("role") not in {"user", "assistant"}
            or not isinstance(turn.get("content"), str)
            for turn in history_list
        ):
            raise HTTPException(422, "history must be a list of user/assistant text turns")
        history_list = [{"role": turn["role"], "content": turn["content"]} for turn in history_list[-HISTORY_MAX:]]

    session_id = session_id or uuid.uuid4().hex
    if history_list is None:
        history_list = HISTORY.get(session_id, [])[-HISTORY_MAX:] or None

    cleanup_tmp()
    image_bytes = [img.file.read() for img in images]
    # Only the refiner gets an RGB preview; downstream tools keep the original bytes.
    first_b64 = _preview_base64(image_bytes[0]) if image_bytes else None

    try:
        refined = refine_query(query, first_b64, history_list[-HISTORY_WINDOW:] if history_list else None)
    except Exception as e:
        raise HTTPException(502, f"Query refinement failed: {e}")

    model = load_lora(refined.get("lora", ""))

    try:
        answer, image_paths = start_conversation_with_images(refined["refined_query"], *image_bytes, model=model, history=history_list)
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Orchestrator failed: {e}")

    # Turn tool-local paths into names the browser can fetch through the file proxy.
    output_images = []
    for image_path in image_paths:
        name = _publish_image(image_path)
        if name:
            output_images.append({"name": name, "url": f"/files/{name}"})
            answer = answer.replace(image_path, f"/files/{name}")  # no raw server paths in the text

    turns = HISTORY[session_id] = list(history_list or [])
    turns += [{"role": "user", "content": query}, {"role": "assistant", "content": answer}]
    del turns[:-HISTORY_MAX]
    if len(HISTORY) > MAX_SESSIONS:
        HISTORY.pop(next(iter(HISTORY)))

    return {"response": answer, "images": output_images, **refined, "model": model, "session_id": session_id}
