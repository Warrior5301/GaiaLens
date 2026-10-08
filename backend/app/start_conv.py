import uuid
from io import BytesIO
from pathlib import Path

from PIL import Image

from backend.app.intelli.common import BASE_MODEL, TMP_DIR, current_model, output_images
from backend.app.intelli.orchestrator.orch import (
    create_orchestrator_agent,
    run_orchestrator,
)

_orchestrators = {}


def get_orchestrator(model: str = BASE_MODEL):
    """Lazily create one shared orchestrator agent per served model (base model or LoRA adapter name)."""
    if model not in _orchestrators:
        token = current_model.set(model)
        try:
            _orchestrators[model] = create_orchestrator_agent()
        finally:
            current_model.reset(token)
    return _orchestrators[model]


def _to_image_path(image) -> str:
    """
    Normalise an image input to a local file path, since the orchestrator's
    sub-agents expect paths on disk.

    Accepts a path (str / Path), a PIL Image, or raw image bytes.
    """
    if isinstance(image, (str, Path)):
        path = Path(image).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Image file not found at {path}")
        return str(path)

    if isinstance(image, (bytes, bytearray)):
        # Keep the original bytes so multiband / high bit-depth data survives
        try:
            with Image.open(BytesIO(image)) as opened:
                ext = (opened.format or "png").lower()
        except Exception as exc:
            raise ValueError("Uploaded file is not a supported image") from exc
        path = TMP_DIR / f"{uuid.uuid4().hex}.{ext}"
        path.write_bytes(image)
        return str(path)

    if isinstance(image, Image.Image):
        path = TMP_DIR / f"{uuid.uuid4().hex}.png"
        image.save(path, format="PNG")
        return str(path)

    raise TypeError(f"Unsupported image type: {type(image).__name__}")


def start_conversation(user_input: str, *images, model: str = BASE_MODEL) -> str:
    """
    Pass a user query and any number of images to the orchestrator.

    Usage:
        start_conversation("What is in this image?", "a.png")
        start_conversation("What changed between these?", "t0.png", "t1.png", "t2.png")
        start_conversation("Describe the region.")  # text only

    Images may be file paths, PIL Images, or raw bytes. A single list/tuple
    of images is also accepted. `model` is the model name served by vLLM (the base
    model, or a hot-loaded LoRA adapter name) used by the orchestrator and all its
    sub-agents. Returns the orchestrator's final response text.
    """
    if not user_input or not user_input.strip():
        raise ValueError("user_input must be a non-empty string")

    if len(images) == 1 and isinstance(images[0], (list, tuple)):
        images = tuple(images[0])

    image_paths = [_to_image_path(img) for img in images]
    agent = get_orchestrator(model)

    # Sub-agents are created lazily during the run and read the model name from this context
    token = current_model.set(model)
    try:
        if not image_paths:
            return run_orchestrator(agent, user_input)

        return run_orchestrator(
            agent,
            user_input,
            image_path=image_paths,
            num_images=len(image_paths),
        )
    finally:
        current_model.reset(token)


def start_conversation_with_images(user_input: str, *images, model: str = BASE_MODEL) -> tuple[str, list[str]]:
    """Like start_conversation, but also returns the paths of the images that tools produced."""
    collected: list[str] = []
    token = output_images.set(collected)
    try:
        answer = start_conversation(user_input, *images, model=model)
    finally:
        output_images.reset(token)
    return answer, collected


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m backend.app.start_conv \"<query>\" [image ...]")
        sys.exit(1)

    start_conversation(sys.argv[1], *sys.argv[2:])
