import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.image_utils import decode_image, draw_overlay, encode_image, mask_to_polygons
from app.model_service import sam3_service
from app.schemas import ObjectResult, SegmentRequest, SegmentResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        sam3_service.load()
    except Exception:
        logger.exception("Failed to load SAM3") 
    yield


app = FastAPI(title="SAM3 Segmentation API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    if not sam3_service.ready:
        raise HTTPException(status_code=503, detail="model not ready")
    return {"status": "ok", "device": sam3_service.device}


@app.post("/run", response_model=SegmentResponse)
def segment(req: SegmentRequest):
    if not sam3_service.ready:
        raise HTTPException(status_code=503, detail="model not ready; check server logs")

    image = decode_image(req.image)

    masks = []
    polys_per_mask = []
    objects = []
    try:
        for prompt in [req.prompt]:
            prompt_masks, boxes, scores = sam3_service.segment(
                image, prompt, req.threshold
            )
            prompt_polys = [mask_to_polygons(m, req.simplify) for m in prompt_masks]
            masks.extend(prompt_masks)
            polys_per_mask.extend(prompt_polys)
            objects.extend(
                ObjectResult(
                    prompt=prompt,
                    score=float(score),
                    box=[float(value) for value in box],
                    polygons=[polygon.astype(int).tolist() for polygon in polygons],
                )
                for score, box, polygons in zip(scores, boxes, prompt_polys)
            )
    except Exception:
        logger.exception("SAM3 inference failed")
        raise HTTPException(status_code=500, detail="inference failed")

    overlay = draw_overlay(image, masks, polys_per_mask)

    return SegmentResponse(
        image=encode_image(overlay),
        count=len(objects),
        objects=objects,
    )


@app.post("/unload")
def unload_model():
    sam3_service.unload()
    return {"status": "unloaded"}
