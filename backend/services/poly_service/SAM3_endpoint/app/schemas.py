from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.config import settings


class SegmentRequest(BaseModel):
    image: str = Field(..., description="Base64 image, optionally a data URL")
    prompt: str = Field(..., min_length=1, description="What to segment, e.g. 'person'")
    threshold: float = Field(settings.DEFAULT_THRESHOLD, ge=0.0, le=1.0, description="Instance score threshold")
    simplify: float = Field(2.0, ge=0.0, description="Polygon simplification epsilon in pixels (0 disables)")

    


class ObjectResult(BaseModel):
    prompt: str
    score: float
    box: list[float]                   # [x0, y0, x1, y1]
    polygons: list[list[list[int]]]    # list of polygons, each a list of [x, y]


class SegmentResponse(BaseModel):
    image: str                         # base64 PNG with masks drawn
    count: int
    objects: list[ObjectResult]
