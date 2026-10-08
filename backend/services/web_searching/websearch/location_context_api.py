"""FastAPI application for the location-context pipeline.

Run from this directory with:
    uvicorn location_context_api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from geoclip_client import (
    MAX_IMAGE_BASE64_CHARS,
    GeoCLIPUnavailableError,
    InvalidImageError,
    decode_image_base64,
)
from location_context_pipeline import OUTPUT_DIR, LocationExplorer


logger = logging.getLogger(__name__)

app = FastAPI(title="Remote Sensing Location Explorer", version="0.1.0")
app.mount("/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")
explorer = LocationExplorer()


class ExploreRequest(BaseModel):
    """Base64 input image and optional satellite-image footprint size."""

    image_base64: str = Field(..., min_length=1, max_length=MAX_IMAGE_BASE64_CHARS)
    imagery_radius_m: float = Field(3000.0, gt=50, le=10000)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/location-explorer")
def location_explorer(request: ExploreRequest) -> dict[str, Any]:
    """Run the pipeline and return coordinates, one Base64 image, and context.

    The ``satellite_images`` and ``web_context`` fields remain lists for
    compatibility with the pipeline response, but this endpoint returns at most
    one item in each. The input may be raw Base64 or a Base64 image data URI.
    """
    try:
        image_bytes = decode_image_base64(request.image_base64)
        result = explorer.explore_image(
            image_bytes,
            imagery_radius_m=request.imagery_radius_m,
        )
    except InvalidImageError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except GeoCLIPUnavailableError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Location exploration failed")
        raise HTTPException(
            status_code=500,
            detail=f"Location exploration failed: {exc}",
        ) from exc

    # Keep the API contract strict even if a future pipeline implementation
    # returns multiple images or search results.
    satellite_images = result.get("satellite_images") or []
    satellite_images = satellite_images[:1]
    if not satellite_images or not satellite_images[0].get("image_base64"):
        raise HTTPException(
            status_code=502,
            detail="The location was found, but no satellite image was available.",
        )
    result["satellite_images"] = satellite_images
    result["web_context"] = (result.get("web_context") or [])[:1]
    # Coordinate search snippets are the concise context returned to clients;
    # do not expose full scraped article text such as Wikipedia history sections.
    for web_result in result["web_context"]:
        web_result["extracted_text"] = None
    return result
