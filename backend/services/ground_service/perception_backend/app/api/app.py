"""FastAPI Perception Control Service exposing explicit model lifecycle and tool APIs."""

from __future__ import annotations

import logging
from typing import Any
from fastapi import Body, FastAPI, HTTPException, Path, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.api.schemas import (
    HealthResponse,
    LoadRequest,
    LoadResponse,
    RunRequest,
    RunResponse,
    SmokeTestRequest,
    SmokeTestResponse,
    UnloadRequest,
    UnloadResponse,
)
from app.core.lifecycle_controller import (
    TOOL_TO_MODEL_MAP,
    lifecycle_controller,
)
from app.core.providers import (
    MockDetectedError,
    ProviderError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

logger = logging.getLogger("app.api.app")

app = FastAPI(
    title="SAC GeoAI Hosting-Agnostic Perception Service",
    description="Explicit model lifecycle orchestration layer with tool-level routing",
    version="2.0.0",
)


@app.exception_handler(ProviderError)
async def provider_error_handler(request: Request, exc: ProviderError) -> JSONResponse:
    logger.warning(f"ProviderError caught ({exc.status_code}): {exc.message}")
    if exc.status_code == 409:
        return JSONResponse(
            status_code=status.HTTP_409_CONFLICT,
            content={"status": "model_not_loaded", "message": exc.message, "detail": exc.message},
        )
    if isinstance(exc, MockDetectedError) or exc.status_code == 503:
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"status": "provider_unavailable", "message": exc.message, "detail": exc.message},
        )
    return JSONResponse(
        status_code=exc.status_code,
        content={"status": "error", "message": exc.message, "detail": exc.message, "details": exc.details},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"status": "validation_error", "message": str(exc), "errors": exc.errors()},
    )


# ============================================================================
# Generic Lifecycle Endpoints
# ============================================================================

@app.post("/load", response_model=LoadResponse, tags=["Lifecycle"])
def load_endpoint(payload: LoadRequest = Body(...)) -> LoadResponse:
    """Request a model provider to load the selected model."""
    try:
        return lifecycle_controller.load_model(payload)
    except ProviderError:
        raise
    except Exception as exc:
        logger.error(f"Error in /load: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/run", response_model=RunResponse, tags=["Inference"])
def run_endpoint(payload: RunRequest = Body(...)) -> RunResponse:
    """Execute inference on an explicitly loaded model."""
    try:
        return lifecycle_controller.run_model(payload)
    except ProviderError:
        raise
    except Exception as exc:
        logger.error(f"Error in /run: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/unload", response_model=UnloadResponse, tags=["Lifecycle"])
def unload_endpoint(payload: UnloadRequest = Body(...)) -> UnloadResponse:
    """Request a model provider to release/unload the model."""
    try:
        return lifecycle_controller.unload_model(payload)
    except ProviderError:
        raise
    except Exception as exc:
        logger.error(f"Error in /unload: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_endpoint() -> HealthResponse:
    """Retrieve service status and current model lifecycle states."""
    return lifecycle_controller.health()


@app.post("/smoke_test", response_model=SmokeTestResponse, tags=["Diagnostics"])
def smoke_test_endpoint(payload: SmokeTestRequest = Body(...)) -> SmokeTestResponse:
    """Validate a complete model lifecycle: LOAD -> RUN default test -> validate -> UNLOAD."""
    try:
        return lifecycle_controller.smoke_test(payload)
    except ProviderError:
        raise
    except Exception as exc:
        logger.error(f"Error in /smoke_test: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ============================================================================
# Tool-Level Routing Endpoints (/tools/{tool_name}/...)
# ============================================================================

@app.post("/tools/{tool_name}/load", response_model=LoadResponse, tags=["Tools"])
def tool_load_endpoint(
    tool_name: str = Path(..., description="Target tool name (e.g. object_detector, segmentation_grounding, image_analyzer)"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> LoadResponse:
    """Tool-specific load route mapping to canonical model ID."""
    canonical_id = TOOL_TO_MODEL_MAP.get(tool_name, tool_name)
    req = LoadRequest(
        model_id=canonical_id,
        run_startup_test=payload.get("run_startup_test", False),
        test_image_base64=payload.get("test_image_base64"),
        test_query=payload.get("test_query"),
        provider=payload.get("provider"),
    )
    return lifecycle_controller.load_model(req)


@app.post("/tools/{tool_name}/run", response_model=RunResponse, tags=["Tools"])
def tool_run_endpoint(
    tool_name: str = Path(..., description="Target tool name"),
    payload: dict[str, Any] = Body(...),
) -> RunResponse:
    """Tool-specific inference route mapping to canonical model ID."""
    canonical_id = TOOL_TO_MODEL_MAP.get(tool_name, tool_name)
    # Build RunRequest adapting tool payload
    req = RunRequest(
        model_id=canonical_id,
        image_base64=payload.get("image_base64", ""),
        query=payload.get("query") or payload.get("text_prompt") or payload.get("question"),
        model_instance_id=payload.get("model_instance_id"),
        pipeline=payload.get("pipeline"),
        target_classes=payload.get("target_classes"),
        parameters=payload.get("parameters") or {},
    )
    return lifecycle_controller.run_model(req)


@app.post("/tools/{tool_name}/unload", response_model=UnloadResponse, tags=["Tools"])
def tool_unload_endpoint(
    tool_name: str = Path(..., description="Target tool name"),
    payload: dict[str, Any] = Body(default_factory=dict),
) -> UnloadResponse:
    """Tool-specific unload route mapping to canonical model ID."""
    canonical_id = TOOL_TO_MODEL_MAP.get(tool_name, tool_name)
    req = UnloadRequest(
        model_id=canonical_id,
        model_instance_id=payload.get("model_instance_id"),
    )
    return lifecycle_controller.unload_model(req)


# ============================================================================
# Backward-Compatible Direct Tool Aliases
# ============================================================================

def _adapt_legacy_payload(tool_name: str, payload: dict[str, Any]) -> RunRequest:
    canonical_id = TOOL_TO_MODEL_MAP.get(tool_name, tool_name)
    img_b64 = payload.get("image_base64", "")
    if not img_b64 and isinstance(payload.get("asset"), dict):
        img_b64 = payload["asset"].get("uri", "")

    return RunRequest(
        model_id=canonical_id,
        image_base64=img_b64,
        query=payload.get("query") or payload.get("text_prompt") or payload.get("question"),
        model_instance_id=payload.get("model_instance_id"),
        pipeline=payload.get("pipeline"),
        target_classes=payload.get("target_classes"),
        parameters=payload.get("parameters") or {},
    )


@app.post("/object_detector", response_model=RunResponse, tags=["Legacy"])
def legacy_object_detector(payload: dict[str, Any] = Body(...)) -> RunResponse:
    req = _adapt_legacy_payload("object_detector", payload)
    return lifecycle_controller.run_model(req)


@app.post("/segmentation_grounding", response_model=RunResponse, tags=["Legacy"])
def legacy_segmentation_grounding(payload: dict[str, Any] = Body(...)) -> RunResponse:
    req = _adapt_legacy_payload("segmentation_grounding", payload)
    return lifecycle_controller.run_model(req)


@app.post("/image_analyzer", response_model=RunResponse, tags=["Legacy"])
def legacy_image_analyzer(payload: dict[str, Any] = Body(...)) -> RunResponse:
    req = _adapt_legacy_payload("image_analyzer", payload)
    return lifecycle_controller.run_model(req)

