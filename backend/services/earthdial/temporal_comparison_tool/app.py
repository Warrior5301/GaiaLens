"""
EarthDial FastAPI Service
=========================

A clean, modular, and framework-agnostic FastAPI application for:
1. Temporal Comparison Tool: Multi-image change detection and temporal reasoning.
2. EarthDial Model: Direct vision-language model inference and conversational VQA.

Run locally or anywhere:
    uvicorn app:app --host 0.0.0.0 --port 8000
    or
    python run.py
"""

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware

# Add package directory to sys.path
package_dir = Path(__file__).resolve().parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

from model import EarthDialModel
from tool import TemporalComparisonTool
from schemas import (
    HealthResponse,
    ImageResolutionInfo,
    ModelChatRequest,
    ModelChatResponse,
    ModelInfoResponse,
    PromptStyle,
    PromptTemplatesResponse,
    TemporalComparisonRequest,
    TemporalComparisonResponse,
)

logger = logging.getLogger("EarthDialAPI")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

# Shared instances
model_instance: Optional[EarthDialModel] = None
tool_instance: Optional[TemporalComparisonTool] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initializes shared model and comparison tool on startup."""
    global model_instance, tool_instance

    checkpoint = os.getenv("EARTHDIAL_CHECKPOINT", "akshaydudhane/EarthDial_4B_RGB")
    device = os.getenv("EARTHDIAL_DEVICE", None)
    mock_mode = os.getenv("EARTHDIAL_MOCK_MODE", "false").lower() in ("true", "1", "yes")
    load_in_8bit = os.getenv("EARTHDIAL_LOAD_IN_8BIT", "false").lower() in ("true", "1", "yes")
    load_in_4bit = os.getenv("EARTHDIAL_LOAD_IN_4BIT", "false").lower() in ("true", "1", "yes")

    logger.info(f"Initializing EarthDial service (checkpoint='{checkpoint}', mock={mock_mode})...")
    model_instance = EarthDialModel(
        checkpoint=checkpoint,
        device=device,
        load_in_8bit=load_in_8bit,
        load_in_4bit=load_in_4bit,
        mock_mode=mock_mode,
        lazy_load=True,
    )
    tool_instance = TemporalComparisonTool(model=model_instance)
    logger.info("Service initialized.")

    yield

    logger.info("Shutting down EarthDial service...")
    model_instance = None
    tool_instance = None


app = FastAPI(
    title="EarthDial Vision-Language Service",
    description="Simple, modular REST API for EarthDial model inference and multi-image temporal comparison.",
    version="2.0.0",
    lifespan=lifespan,
)

# Standard CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# General / Health Endpoints
# ============================================================================

@app.get("/", tags=["General"])
async def root():
    """Service overview and quick-start links."""
    return {
        "service": "EarthDial Vision-Language Service",
        "version": "2.0.0",
        "docs": "/docs",
        "endpoints": {
            "temporal_comparison": {
                "compare_json": "POST /compare",
                "compare_upload": "POST /compare/upload",
                "templates": "GET /templates"
            },
            "model": {
                "chat_vqa": "POST /model/chat",
                "info": "GET /model/info"
            },
            "monitoring": {
                "health": "GET /health"
            }
        }
    }


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """System and GPU acceleration health diagnostics."""
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model service not initialized.")

    info = model_instance.get_info()
    return HealthResponse(
        status="healthy",
        model_loaded=info["model_loaded"],
        model_path=info["model_path"],
        device=info["device"],
        cuda_available=info["cuda_available"],
        gpu_name=info["gpu_name"],
        gpu_memory_allocated_gb=info["gpu_memory_allocated_gb"],
    )


# ============================================================================
# Temporal Comparison Tool Endpoints
# ============================================================================

@app.post("/compare", response_model=TemporalComparisonResponse, tags=["Temporal Comparison Tool"])
async def compare_images(request: TemporalComparisonRequest):
    """
    Compare 2 or more remote-sensing images using file paths, URLs, or base64 strings.
    """
    global tool_instance
    if tool_instance is None:
        raise HTTPException(status_code=503, detail="Tool service unavailable.")

    try:
        return tool_instance.compare_from_request(request)
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Error in /compare: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(exc)}")


@app.post("/compare/upload", response_model=TemporalComparisonResponse, tags=["Temporal Comparison Tool"])
async def compare_uploaded_files(
    files: List[UploadFile] = File(..., description="2 or more image files uploaded as multipart/form-data"),
    query: str = Form(
        "Analyze the images and identify all semantic and structural changes between the observations.",
        description="The comparison question or instruction."
    ),
    prompt_style: PromptStyle = Form(
        PromptStyle.EVALUATION,
        description="Prompt style formatting (evaluation or conversational)."
    ),
    task_token: Optional[str] = Form(
        "[changedet] [hr_rgb_temp_0.5]",
        description="Special task prefix token for EarthDial."
    ),
    temperature: float = Form(0.0, description="Sampling temperature (0.0 = greedy)."),
    max_new_tokens: int = Form(256, description="Max generated tokens."),
    dynamic_image_size: bool = Form(True, description="Enable dynamic multi-crop tiling."),
):
    """
    Upload 2 or more image files directly via multipart/form-data for temporal comparison.
    """
    global tool_instance
    if tool_instance is None:
        raise HTTPException(status_code=503, detail="Tool service unavailable.")

    if len(files) < 2:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Temporal comparison requires at least 2 images, but received {len(files)}."
        )

    try:
        image_bytes = []
        for file in files:
            content = await file.read()
            if not content:
                raise HTTPException(status_code=400, detail=f"Uploaded file '{file.filename}' is empty.")
            image_bytes.append(content)

        return tool_instance.compare(
            images=image_bytes,
            query=query,
            prompt_style=prompt_style,
            task_token=task_token,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
            dynamic_image_size=dynamic_image_size,
        )
    except HTTPException:
        raise
    except ValueError as ve:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
    except Exception as exc:
        logger.error(f"Error in /compare/upload: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(exc)}")


@app.get("/templates", response_model=PromptTemplatesResponse, tags=["Temporal Comparison Tool"])
async def get_prompt_templates():
    """Returns recommended prompt templates for various remote-sensing domains."""
    return PromptTemplatesResponse(templates=TemporalComparisonTool.get_templates())


# ============================================================================
# Direct Model Endpoints
# ============================================================================

@app.post("/model/chat", response_model=ModelChatResponse, tags=["EarthDial Model"])
async def model_chat(request: ModelChatRequest):
    """
    Direct prompt/VQA inference with EarthDial on 0, 1, or multiple images.
    """
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model service unavailable.")

    try:
        t0 = time.time()
        pil_images = []
        if request.images:
            for img_input in request.images:
                pil_images.append(TemporalComparisonTool.load_image(img_input))

        # If images are provided but user didn't include <image> tokens in prompt, prepend them
        final_prompt = request.prompt
        if pil_images and "<image>" not in final_prompt:
            prefix = "".join(f"Image-{i + 1}: <image>\n" for i in range(len(pil_images)))
            final_prompt = f"{prefix}{final_prompt}"

        generated_text, meta_list, device_used = model_instance.generate(
            images=pil_images if pil_images else None,
            prompt=final_prompt,
            temperature=request.temperature,
            max_new_tokens=request.max_new_tokens,
            top_p=request.top_p,
            num_beams=request.num_beams,
            dynamic_image_size=request.dynamic_image_size,
            max_tiles_per_image=request.max_tiles_per_image,
        )
        elapsed = round(time.time() - t0, 3)

        image_info = [
            ImageResolutionInfo(
                index=m["index"],
                width=m["width"],
                height=m["height"],
                num_tiles=m.get("num_tiles", 1)
            )
            for m in meta_list
        ]

        return ModelChatResponse(
            success=True,
            prompt=final_prompt,
            response=generated_text,
            num_images=len(pil_images),
            image_metadata=image_info,
            execution_time_seconds=elapsed,
            device=device_used,
            model_name_or_path=model_instance.checkpoint,
        )
    except Exception as exc:
        logger.error(f"Error in /model/chat: {exc}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Inference failed: {str(exc)}")


@app.get("/model/info", response_model=ModelInfoResponse, tags=["EarthDial Model"])
async def model_info():
    """Retrieve detailed information about the EarthDial model and hardware acceleration."""
    global model_instance
    if model_instance is None:
        raise HTTPException(status_code=503, detail="Model service unavailable.")

    info = model_instance.get_info()
    return ModelInfoResponse(
        model_path=info["model_path"],
        device=info["device"],
        model_loaded=info["model_loaded"],
        cuda_available=info["cuda_available"],
        gpu_name=info["gpu_name"],
        gpu_memory_allocated_gb=info["gpu_memory_allocated_gb"],
        image_size=info["image_size"],
    )
