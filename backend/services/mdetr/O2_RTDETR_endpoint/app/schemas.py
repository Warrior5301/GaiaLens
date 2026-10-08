from pydantic import BaseModel, ConfigDict, Field

from app.config import DEFAULT_SCORE_THRESHOLD


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    image: str = Field(
        ...,
        min_length=1,
        description="Base64-encoded JPEG/PNG, optionally a data:image/...;base64 URI",
    )
    score_threshold: float = Field(
        DEFAULT_SCORE_THRESHOLD, ge=0.0, le=1.0,
        description="Hide detections with a lower confidence score",
    )


class Detection(BaseModel):
    label: str
    score: float
    # O2-RTDETR's native rotated-box order: [cx, cy, width, height, angle].
    box: list[float]


class PredictResponse(BaseModel):
    image: str = Field(..., description="Base64-encoded annotated PNG")
    media_type: str = "image/png"
    count: int
    detections: list[Detection]
