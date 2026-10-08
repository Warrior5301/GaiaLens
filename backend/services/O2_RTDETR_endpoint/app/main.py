import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.image_utils import decode_image, encode_png
from app.model_service import o2_rtdetr_service
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("o2_rtdetr_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        o2_rtdetr_service.load()
    except Exception:
        logger.exception("Failed to load O2-RTDETR")
    yield
    o2_rtdetr_service.unload()


app = FastAPI(
    title="O2-RTDETR Remote Sensing Detection API",
    version="1.0.0",
    description="Base64 image in; base64 annotated PNG and oriented detections out.",
    lifespan=lifespan,
)


@app.get("/health")
def health():
    if not o2_rtdetr_service.ready:
        raise HTTPException(status_code=503, detail="model not ready; check server logs")
    return {"status": "ok", "device": o2_rtdetr_service.device}


@app.post("/run", response_model=PredictResponse)
def run_inference(req: PredictRequest):
    if not o2_rtdetr_service.ready:
        raise HTTPException(status_code=503, detail="model not ready; check server logs")

    image = decode_image(req.image)
    try:
        rendered_rgb, detections = o2_rtdetr_service.predict(image, req.score_threshold)
    except Exception as exc:
        logger.exception("O2-RTDETR inference failed")
        raise HTTPException(status_code=500, detail="inference failed") from exc

    return PredictResponse(
        image=encode_png(rendered_rgb),
        count=len(detections),
        detections=detections,
    )
