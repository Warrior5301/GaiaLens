"""Simple FastAPI wrapper around the deployed TerraScope Modal function.

Usage:
    1. Deploy the Modal app: modal deploy modal_app.py
    2. Start this API:       uvicorn api:app --reload --host 0.0.0.0 --port 8000

Then POST to http://localhost:8000/run with JSON:
    {
        "image_base64": "<base64 string>",
        "prompt": "Describe this satellite image.",
        "compare_image_base64": null
    }
"""

import base64
import re

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import modal
from pydantic import BaseModel

app = FastAPI(title="TerraScope API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    image_base64: str
    prompt: str = "Describe this satellite image."
    compare_image_base64: str | None = None


class AnalyzeResponse(BaseModel):
    answer: str
    overlay_base64: str | None = None
    overlay_mime: str | None = None
    coverage_percent: list[float] | None = None


# ── Helpers ──────────────────────────────────────────────────────────────────

def _decode_b64(value: str, field: str) -> bytes:
    """Decode a raw or data-URL base64 string to bytes."""
    encoded = value.strip()
    if not encoded:
        raise HTTPException(422, f"'{field}' is empty.")
    if encoded.startswith("data:"):
        _, _, encoded = encoded.partition(",")
    try:
        return base64.b64decode(encoded, validate=True)
    except Exception as exc:
        raise HTTPException(422, f"'{field}' is not valid base64.") from exc


def _clean_answer(raw: str) -> str:
    """Strip TerraScope markup tags from the model answer."""
    text = raw.replace("[SEG]", "").replace("<|im_end|>", "").strip()
    m = re.search(r"<answer>\s*(.*?)\s*</answer>", text, re.I | re.S)
    if m:
        text = m.group(1).strip()
    text = re.sub(r"</?(?:think|answer|p)>", "", text, flags=re.I).strip()
    return text


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/run", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    """Send an image + prompt to TerraScope via Modal and return the result."""
    image_bytes = _decode_b64(req.image_base64, "image_base64")

    compare_bytes = None
    if req.compare_image_base64:
        compare_bytes = _decode_b64(req.compare_image_base64, "compare_image_base64")

    try:
        run_test = modal.Function.from_name("terrascope-bf16-test", "run_test")
        answer_raw, overlay_bytes, coverages = run_test.remote(
            image_bytes, req.prompt, compare_bytes
        )
    except Exception as exc:
        raise HTTPException(500, f"Modal inference failed: {exc}") from exc

    answer = _clean_answer(str(answer_raw))

    # If prompt asks about area/coverage, append the info
    if coverages and re.search(
        r"\b(area|proportion|percentage|percent|coverage|how much)\b",
        req.prompt,
        re.I,
    ):
        answer += (
            f" Estimated target coverage: {coverages[0]:.1f}% of image pixels."
            " This is pixel coverage, not a real-world area measurement."
        )

    result = AnalyzeResponse(answer=answer)

    if overlay_bytes:
        result.overlay_base64 = base64.b64encode(overlay_bytes).decode("ascii")
        result.overlay_mime = "image/jpeg"
        result.coverage_percent = [round(v, 1) for v in coverages]

    return result
