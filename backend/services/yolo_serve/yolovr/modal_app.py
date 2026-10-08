"""Deploy a YOLO-World image + phrase inference endpoint on Modal.

Deploy from this directory with:
    modal deploy modal_app.py

The fine-tuned checkpoint must be copied here as ``best.pt`` first.
"""

import base64
import io
from pathlib import Path

import modal


ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / "best.pt"
APP_NAME = "yolovr-inference"

model_image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install(
        "torch==2.3.1",
        "torchvision==0.18.1",
        "ultralytics==8.4.174",
        "clip @ https://github.com/openai/CLIP/archive/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1.zip",
        "numpy<2",
        "pillow",
        "opencv-python-headless",
        "fastapi",
    )
    .add_local_file(str(CHECKPOINT), remote_path="/model/best.pt")
)
app = modal.App(APP_NAME)


@app.function(
    image=model_image,
    gpu="A10",
    timeout=180,
    scaledown_window=300,
)
@modal.asgi_app()
def web_app():
    """Serve direct base64 image + prompt inference over HTTP."""
    import torch
    from fastapi import FastAPI, HTTPException
    from PIL import Image, ImageDraw
    from pydantic import BaseModel
    from ultralytics import YOLOWorld

    print("Loading YOLO-World checkpoint...", flush=True)
    model = YOLOWorld("/model/best.pt")
    device = "0" if torch.cuda.is_available() else "cpu"
    api = FastAPI(title="YOLO-World VRSBench Inference")

    class PredictRequest(BaseModel):
        image_base64: str
        prompt: str

    @api.get("/health")
    async def health():
        return {"status": "ok", "device": device}

    @api.post("/predict")
    async def predict(request: PredictRequest):
        if not request.prompt.strip():
            raise HTTPException(status_code=422, detail="prompt cannot be empty")
        encoded = request.image_base64.strip()
        if encoded.startswith("data:"):
            header, separator, encoded = encoded.partition(",")
            if not separator or "base64" not in header.lower():
                raise HTTPException(status_code=422, detail="image data URL must be base64 encoded")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except Exception as exc:
            raise HTTPException(status_code=422, detail="image_base64 is not valid base64") from exc
        if not raw:
            raise HTTPException(status_code=422, detail="decoded image is empty")
        try:
            with Image.open(io.BytesIO(raw)) as source:
                pil_image = source.convert("RGB")
        except Exception as exc:
            raise HTTPException(status_code=422, detail="could not read decoded image") from exc

        prompt = request.prompt.strip()
        model.set_classes([prompt])
        print(f"Running inference for prompt: {prompt!r}", flush=True)
        predictions = model.predict(
            source=pil_image,
            conf=0.01,
            device=device,
            verbose=False,
        )
        boxes = predictions[0].boxes
        draw = ImageDraw.Draw(pil_image)
        if len(boxes):
            scores = boxes.conf.cpu().tolist()
            coordinates = boxes.xyxy.cpu().tolist()
            order = sorted(range(len(scores)), key=lambda idx: scores[idx], reverse=True)
            for idx in order:
                box = [round(float(v), 2) for v in coordinates[idx]]
                score = round(float(scores[idx]), 4)
                draw.rectangle(box, outline=(0, 255, 0), width=max(2, pil_image.width // 300))
                draw.text((box[0], max(0, box[1] - 18)), f"{prompt} ({score:.2f})", fill=(0, 255, 0))

        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        image_result = base64.b64encode(buffer.getvalue()).decode("ascii")
        return {"image_base64": image_result}

    return api
