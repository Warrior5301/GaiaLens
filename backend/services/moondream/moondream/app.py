"""FastAPI service for Moondream 3 query refinement and LoRA routing."""

from __future__ import annotations

import base64
import binascii
import gc
import io
import json
import logging
import os
import shutil
import threading
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, HTTPException
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger("moondream.api")

MODEL_ID = os.getenv("MOONDREAM_MODEL_ID", "moondream/moondream3-preview")
MODEL_REVISION = os.getenv(
    "MOONDREAM_MODEL_REVISION", "5112966d1a723413b1c9a1e8bea272b72e647b35"
)
CACHE_DIR = Path(os.getenv("MOONDREAM_CACHE_DIR", "/cache/huggingface")).expanduser()
MODULES_CACHE_DIR = os.getenv(
    "HF_MODULES_CACHE", "/tmp/moondream-transformers-modules"
)
MAX_IMAGE_BYTES = 12 * 1024 * 1024
LORAS = ("Grounding", "VQA", "Captioning", "SAR","IR")

os.environ.setdefault("HF_HOME", str(CACHE_DIR))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_DIR / "hub"))
os.environ.setdefault("HF_MODULES_CACHE", MODULES_CACHE_DIR)
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

app = FastAPI(
    title="Moondream3 Query Refiner",
    version="2.0.0",
    description=(
        "Refines Earth-observation queries and routes them to a downstream LoRA. "
        "Optional images are accepted as base64."
    ),
)

_model: Any = None
_lock = threading.Lock()


class Turn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    role: str = Field(min_length=1, max_length=40)
    content: str = Field(max_length=8000)


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=8000)
    image_base64: str | None = None
    history: list[Turn] = Field(default_factory=list)


SYSTEM_PROMPT = """You are the query-understanding and query-planning module of GaiaLens, an Earth Observation / remote-sensing assistant.

A downstream vision-language model (Qwen3-VL) and specialized tools execute the plan and answer the user. You do NOT answer the user's question yourself. Your job is to understand the request, resolve context, select the appropriate model adapter, and create a concise execution plan for the downstream system.

Given:
- conversation history (may be empty),
- the current user query,
- optionally one or more images,

perform the following in order:

1. HISTORY CHECK
Decide whether the current query depends on the conversation history.

The query is related to history if it:
- contains references such as "it", "they", "those", "that", "the previous image", etc.;
- asks "what about...", "now do the same...", "compare this with the previous one...", etc.;
- refers to an earlier result, observation, object, location, image, or conclusion;
- requires information established earlier in the conversation to understand the request.

If the query is self-contained, starts a new topic, or the history is empty, set related_to_history to false.

2. INTENT
State in one short sentence what the user actually wants.

Examples:
- "locate all ships in the image"
- "count buildings in the image"
- "compare land-cover changes between two images"
- "identify flooded areas in the SAR image"
- "describe the overall scene"
- "determine whether new construction is present"

Do not answer the intent.

3. REFINED QUERY
Rewrite the user's query as one clear, self-contained instruction for the downstream model.

Rules:
- If related_to_history is true, resolve references using the conversation history.
- If related_to_history is false, use ONLY the current query and provided image(s); do not leak irrelevant history.
- Preserve the user's actual intent.
- Remove ambiguity, filler, and conversational wording.
- Do not invent facts, locations, objects, dates, sensor types, or constraints.
- Do not answer the question.
- Make the instruction precise enough for another model to execute.

4. TASK DECOMPOSITION
Determine whether the request requires multiple reasoning or vision steps.

If the task is simple, create a single-step plan.

If the task is complex, decompose it into the minimum number of ordered steps required to answer the request.

Examples:

User: "Find the buildings that were newly constructed between these two images."

Plan:
1. Identify and localize buildings in the earlier image.
2. Identify and localize buildings in the later image.
3. Compare the detected buildings spatially.
4. Identify buildings present in the later image but absent from the earlier image.
5. Report the newly constructed buildings and their locations.

User: "Are there more ships near the harbor than near the airport?"

Plan:
1. Locate the harbor and airport regions.
2. Detect ships in the harbor region.
3. Detect ships in the airport region.
4. Count the ships in each region.
5. Compare the counts.

The plan must describe actions, not conclusions.

5. TOOL / CAPABILITY PLANNING
For every plan step, determine what capability is required.

Possible capabilities include:
- "visual_understanding" — understand image content, scene, objects, relationships, or attributes.
- "object_detection" — locate objects and return bounding boxes.
- "grounding" — locate an object described by natural language.
- "segmentation" — produce precise object/region masks or polygons.
- "counting" — count instances of objects.
- "image_comparison" — compare two or more images.
- "change_detection" — identify spatial/visual changes between images.
- "spatial_reasoning" — reason about relative positions, regions, distances, or relationships.
- "ocr" — read text from imagery.
- "classification" — classify an image, region, or object.
- "captioning" — describe or summarize the scene.
- "sar_ir_analysis" — interpret SAR, radar, infrared, thermal, or multispectral imagery.
- "reasoning" — perform comparison, inference, yes/no reasoning, or multi-step analysis.

Only include capabilities that are actually necessary.

6. EXECUTION ORDER
Specify the order in which the capabilities should be executed.

Dependencies must be respected.

For example:
- detection must occur before counting if objects need to be individually detected;
- segmentation must occur before area calculation if precise region boundaries are required;
- detection on two images must occur before change detection;
- OCR must occur before reasoning about text contained in the image.

If steps are independent, they may be marked as parallelizable.

7. LORA SELECTION
Choose exactly one adapter for the downstream model:

- "Grounding":
  locate, detect, count-with-position, bounding boxes, segmentation, "where is", spatial localization.

- "VQA":
  specific questions, counting, reasoning, comparison, yes/no, classification, analysis, or anything not covered below.

- "Captioning":
  describe, caption, or summarize the whole scene/image.

- "SAR":
  the image or query concerns SAR
- "IR":
  the image or query concerns infrared.

8. OUTPUT PLAN
Create a concise ordered execution plan.

Each step should contain:
- what action should be performed,
- which capability/tool is required,
- what the step should produce for the next step.

Do not perform the steps yourself.

9. OUTPUT FORMAT
Output ONLY a valid JSON object.
Do not output markdown, explanations, comments, or extra text.

Use exactly this structure:

{
  "related_to_history": <true|false>,
  "intent": "<short description of what the user wants>",
  "refined_query": "<self-contained instruction for the downstream model>",
  "lora": "<Grounding|VQA|Captioning|SAR/IR>",
  "plan": [
    {
      "step": 1,
      "action": "<what to do>",
      "capability": "<required capability>",
      "output": "<what this step should produce>"
    }
  ]
}

The plan must be minimal but sufficient to solve the user's request.
Do not add unnecessary steps.
Do not answer the user's question.
Do not invent information not present in the query, history, or image.
"""

def _load_model() -> Any:
    global _model
    if _model is not None:
        return _model
    if not torch.cuda.is_available():
        raise RuntimeError("Moondream 3 requires a CUDA GPU")

    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    model_path = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=str(CACHE_DIR),
    )
    if not (Path(model_path) / "lora.py").is_file():
        raise FileNotFoundError(
            f"Downloaded Moondream snapshot at {model_path} is missing lora.py"
        )
    _stage_remote_code(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map={"": "cuda"},
        low_cpu_mem_usage=True,
    ).eval()

    # Moondream 3 uses FlexAttention. Its compile method performs the model's
    # required setup/warmup; if compilation is unsupported, retain the loaded
    # model and log the fallback so inference errors remain visible.
    compile_model = getattr(model, "compile", None)
    if callable(compile_model):
        try:
            compile_model()
        except Exception:
            logger.exception("Moondream compile failed; continuing without compilation")

    _model = model
    logger.info("Loaded %s at revision %s", MODEL_ID, MODEL_REVISION)
    return _model


def _stage_remote_code(model_path: str) -> None:
    """Copy all sibling model modules into Transformers' dynamic-code package.

    The model's remote Python files use relative imports. Staging the complete
    set at once prevents Transformers from encountering a partially populated
    module cache while it recursively resolves those imports.
    """
    from transformers.utils import HF_MODULES_CACHE

    snapshot_dir = Path(model_path)
    module_dir = (
        Path(HF_MODULES_CACHE)
        / "transformers_modules"
        / snapshot_dir.name
    )
    module_dir.mkdir(parents=True, exist_ok=True)
    (Path(HF_MODULES_CACHE) / "__init__.py").touch(exist_ok=True)
    package_dir = module_dir.parent
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").touch(exist_ok=True)
    (module_dir / "__init__.py").touch(exist_ok=True)

    source_files = list(snapshot_dir.glob("*.py"))
    if not source_files:
        raise FileNotFoundError(f"No remote Python modules found in {snapshot_dir}")
    for source in source_files:
        shutil.copy2(source, module_dir / source.name)
    logger.info(
        "Staged %d Moondream remote-code files in %s",
        len(source_files),
        module_dir,
    )


def _decode_image(encoded: str) -> Image.Image:
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("image_base64 must be raw base64 or a base64 data URL")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image_base64 is not valid base64") from exc
    if not raw:
        raise ValueError("image_base64 is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError("image_base64 exceeds the 12 MiB image limit")
    try:
        with Image.open(io.BytesIO(raw)) as opened:
            opened.load()
            return opened.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("image_base64 is not a supported image") from exc


def _extract_json_object(raw: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for index, character in enumerate(raw):
        if character != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(raw[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError(f"Model did not return a JSON object: {raw[:300]}")


def _build_prompt(req: RunRequest, has_image: bool) -> tuple[str, str]:
    history = "".join(
        f"{turn.role}: {turn.content}\n"
        for turn in req.history
    )
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Conversation history:\n{history or '(none)'}\n"
        f"Image attached: {'yes' if has_image else 'no'}\n"
        f"Current query: {req.query.strip()}\n\nJSON:"
    )
    return prompt, history


def _normalize_result(raw: str, query: str, has_history: bool) -> dict[str, Any]:
    result = _extract_json_object(raw)
    lora = result.get("lora")
    if lora not in LORAS:
        lora = "VQA"
    intent = result.get("intent")
    refined_query = result.get("refined_query")
    return {
        "related_to_history": bool(result.get("related_to_history")) and has_history,
        "intent": intent.strip() if isinstance(intent, str) else "",
        "refined_query": refined_query.strip()
        if isinstance(refined_query, str) and refined_query.strip()
        else query,
        "lora": lora,
    }


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": "Moondream3 Query Refiner",
        "health": "/health",
        "load": "POST /load",
        "run": "POST /run",
        "unload": "POST /unload",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return {"status": "ok", "model": MODEL_ID, "loaded": _model is not None}


@app.post("/load")
def load() -> dict[str, str]:
    try:
        with _lock:
            _load_model()
        return {"status": "loaded", "model": MODEL_ID}
    except Exception as exc:
        logger.exception("Moondream model load failed")
        raise HTTPException(
            status_code=500,
            detail=f"Model load failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/unload")
def unload() -> dict[str, str]:
    global _model
    with _lock:
        _model = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {"status": "unloaded"}


@app.post("/run")
def run(req: RunRequest) -> dict[str, Any]:
    if not req.query.strip():
        raise HTTPException(status_code=422, detail="query cannot be blank")

    try:
        image = _decode_image(req.image_base64) if req.image_base64 else None
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    question, history = _build_prompt(req, image is not None)
    with _lock:
        if _model is None:
            raise HTTPException(status_code=409, detail="Model not loaded; call /load first")
        try:
            with torch.inference_mode():
                response = _model.query(
                    image=image,
                    question=question,
                    reasoning=False,
                    settings={"temperature": 0.0, "max_tokens": 512},
                )
        except Exception as exc:
            logger.exception("Moondream inference failed")
            raise HTTPException(
                status_code=500,
                detail=f"Inference failed: {type(exc).__name__}: {exc}",
            ) from exc

    raw = response.get("answer", "")
    if not isinstance(raw, str) or not raw.strip():
        raise HTTPException(status_code=502, detail="Moondream returned an empty answer")
    try:
        return _normalize_result(raw, req.query, bool(history))
    except ValueError as exc:
        logger.error("Moondream returned invalid structured output: %s", raw[:500])
        raise HTTPException(status_code=502, detail=str(exc)) from exc
