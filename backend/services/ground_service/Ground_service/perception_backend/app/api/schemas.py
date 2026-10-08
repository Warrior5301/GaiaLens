"""Standardized Pydantic schemas for Model Lifecycle and Inference APIs."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field

from sac_shared_contracts.common import Detection, StrictModel
from sac_shared_contracts.perception import GroundingResult


class LoadRequest(StrictModel):
    """Request schema for explicit model loading."""
    model_id: str = Field(description="Target model identifier (e.g. remote_sam_sam3_v2, yolo_obb_v1, earthmind_v1)")
    run_startup_test: bool = Field(default=False, description="Whether to execute a startup validation inference test")
    test_image_base64: Optional[str] = Field(default=None, description="Optional custom image for startup test")
    test_query: Optional[str] = Field(default=None, description="Optional custom query for startup test")
    provider: Optional[str] = Field(default=None, description="Optional explicit provider override (http, modal, local)")


class LoadResponse(StrictModel):
    """Response schema for model load operation."""
    status: str = Field(default="loaded", description="Load status")
    model_id: str = Field(description="Loaded model identifier")
    model_instance_id: str = Field(description="Unique runtime session instance identifier")
    provider: str = Field(description="Provider fulfilling execution")
    real_model: bool = Field(default=True, description="Strict guarantee: True for real model checkpoints")
    load_latency_ms: float = Field(description="Model load duration in milliseconds")
    device: Optional[str] = Field(default=None, description="Hardware compute device (cuda / cpu)")
    dtype: Optional[str] = Field(default=None, description="Tensor precision (fp16, bfloat16, fp32)")
    checkpoint: Optional[str] = Field(default=None, description="Weights checkpoint path or URI")
    smoke_test: Optional[dict[str, Any]] = Field(default=None, description="Startup smoke test results if requested")


class RunRequest(StrictModel):
    """Standardized inference request schema across all perception tools."""
    model_id: str = Field(description="Target model identifier")
    image_base64: str = Field(description="Raw base64 or data:image/...;base64 encoded image string")
    query: Optional[str] = Field(default=None, description="Natural language prompt or referring expression")
    model_instance_id: Optional[str] = Field(default=None, description="Optional target session instance identifier")
    pipeline: Optional[str] = Field(default=None, description="Optional model pipeline variant")
    target_classes: Optional[list[str]] = Field(default=None, description="Explicit target object classes (for detectors)")
    image_mime_type: Optional[str] = Field(default="image/png", description="Image MIME type (image/png, image/jpeg)")
    parameters: Optional[dict[str, Any]] = Field(default_factory=dict, description="Hyperparameters and thresholds")


class RunResponse(StrictModel):
    """Standardized inference response schema adhering to shared perception contracts."""
    status: str = Field(default="success", description="Execution status")
    model_id: str = Field(description="Executed model identifier")
    model_instance_id: Optional[str] = Field(default=None, description="Runtime instance identifier")
    provider: str = Field(description="Provider that handled inference")
    real_model: bool = Field(default=True, description="Strict real model provenance")
    latency_ms: float = Field(description="Inference duration in milliseconds")
    image_dimensions: Optional[list[int]] = Field(default=None, description="Processed image [width, height]")
    detections: Optional[list[Detection]] = Field(default=None, description="Localized bounding box detections")
    count: Optional[int] = Field(default=None, description="Exact detection count cardinality")
    groundings: Optional[list[GroundingResult]] = Field(default=None, description="Pixel-level segmentation groundings")
    groundings_count: Optional[int] = Field(default=None, description="Cardinality of grounded referents")
    response: Optional[dict[str, Any]] = Field(default=None, description="Semantic VLM captioning/VQA response")
    model_metadata: Optional[dict[str, Any]] = Field(default=None, description="Detailed model execution metadata")
    processing_metrics: Optional[dict[str, Any]] = Field(default=None, description="Internal processing metrics")


class UnloadRequest(StrictModel):
    """Request schema for explicit model unload."""
    model_id: Optional[str] = Field(default=None, description="Model identifier to unload")
    model_instance_id: Optional[str] = Field(default=None, description="Specific instance identifier to unload")


class UnloadResponse(StrictModel):
    """Response schema for model unload operation."""
    status: str = Field(default="unloaded", description="Unload status")
    model_id: Optional[str] = Field(default=None, description="Unloaded model identifier")
    model_instance_id: Optional[str] = Field(default=None, description="Unloaded instance identifier")
    unload_latency_ms: float = Field(description="Unload latency in milliseconds")
    gpu_memory_released_mb: Optional[float] = Field(default=None, description="GPU memory freed in MB if local")
    memory_release: Optional[str] = Field(default=None, description="Memory release provenance")


class SmokeTestRequest(StrictModel):
    """Request schema for automated full lifecycle validation."""
    model_id: str = Field(description="Target model identifier")
    image_base64: Optional[str] = Field(default=None, description="Optional custom image for smoke test")
    query: Optional[str] = Field(default=None, description="Optional custom query for smoke test")


class SmokeTestResponse(StrictModel):
    """Response schema from complete load -> run -> unload validation test."""
    status: str = Field(default="passed", description="Validation outcome (passed / failed)")
    model_id: str = Field(description="Tested model identifier")
    provider: str = Field(description="Provider evaluated")
    real_model: bool = Field(default=True, description="Strict real model confirmation")
    load: dict[str, Any] = Field(description="Load step metrics")
    run: dict[str, Any] = Field(description="Run step metrics")
    unload: dict[str, Any] = Field(description="Unload step metrics")
    result: dict[str, Any] = Field(description="Inference result snippet")


class HealthResponse(StrictModel):
    """Response schema for system and model registry health query."""
    status: str = Field(default="healthy", description="Overall service status")
    service: str = Field(default="sac-geoai-perception", description="Service name")
    models: dict[str, dict[str, Any]] = Field(description="Lifecycle states for all configured models")
