"""
Pydantic schemas and data models for EarthDial Temporal Comparison Tool & Model API.
"""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class PromptStyle(str, Enum):
    """Supported prompt styling for EarthDial temporal queries."""
    EVALUATION = "evaluation"        # [changedet] [hr_rgb_temp_0.5] <image> <image> \n {query}
    CONVERSATIONAL = "conversational"  # Image-1: <image>\nImage-2: <image>\n{query}
    RAW = "raw"                      # Uses user-supplied prompt string as-is


class ImageResolutionInfo(BaseModel):
    """Metadata about an input image."""
    index: int
    width: int
    height: int
    format: Optional[str] = None
    num_tiles: int = 1


# ============================================================================
# 1. Temporal Comparison Tool Schemas
# ============================================================================

class TemporalComparisonRequest(BaseModel):
    """Request payload for temporal image comparison (requires >= 2 images)."""
    images: List[str] = Field(
        ...,
        min_length=2,
        description="List of 2 or more images as local file paths, base64 data URIs, or accessible URLs."
    )
    query: str = Field(
        default="Analyze the images and identify all semantic and structural changes between the observations.",
        description="The comparison question or instruction for EarthDial."
    )
    prompt_style: PromptStyle = Field(
        default=PromptStyle.EVALUATION,
        description="Prompt style formatting for multi-image tokens."
    )
    task_token: Optional[str] = Field(
        default="[changedet] [hr_rgb_temp_0.5]",
        description="Special task prefix token used during EarthDial training."
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum new tokens to generate."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0.0 = greedy decoding (recommended for evaluation)."
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling probability."
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of beams for beam search."
    )
    dynamic_image_size: bool = Field(
        default=True,
        description="Whether to use multi-crop dynamic resolution tiling."
    )
    max_tiles_per_image: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Max dynamic image tiles per image."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "images": [
                    "test_data/satellite-image-t1.png",
                    "test_data/satellite-image-t2.png"
                ],
                "query": "Are there any semantic changes detected in the provided images?",
                "prompt_style": "evaluation",
                "temperature": 0.0,
                "max_new_tokens": 150
            }
        }
    }


class TemporalComparisonResponse(BaseModel):
    """Response returned by the temporal comparison tool."""
    success: bool = Field(default=True, description="Whether the inference succeeded.")
    query: str = Field(..., description="The user's query.")
    formatted_prompt: str = Field(..., description="The prompt with image tokens sent to the model.")
    response: str = Field(..., description="EarthDial generated temporal comparison analysis.")
    num_images: int = Field(..., description="Number of images compared.")
    image_metadata: List[ImageResolutionInfo] = Field(default_factory=list, description="Metadata for each input image.")
    execution_time_seconds: float = Field(..., description="Latency in seconds.")
    device: str = Field(..., description="Execution device (e.g. cuda, cpu, mock).")
    model_name_or_path: str = Field(..., description="Model identifier or checkpoint path.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional debug/generation metadata.")


# ============================================================================
# 2. Direct Model Schemas (Single/Multi-Image or Text VQA)
# ============================================================================

class ModelChatRequest(BaseModel):
    """Request payload for direct interaction with the EarthDial model."""
    prompt: str = Field(
        ...,
        description="Text prompt or question to ask the model. If images are provided, you can reference them or they will be prepended."
    )
    images: Optional[List[str]] = Field(
        default_factory=list,
        description="Optional list of image file paths, URLs, or base64 data URIs (0, 1, or more)."
    )
    max_new_tokens: int = Field(
        default=256,
        ge=1,
        le=2048,
        description="Maximum new tokens to generate."
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (0.0 = greedy)."
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Top-p nucleus sampling probability."
    )
    num_beams: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of beams for beam search."
    )
    dynamic_image_size: bool = Field(
        default=True,
        description="Whether to use multi-crop dynamic resolution tiling."
    )
    max_tiles_per_image: int = Field(
        default=6,
        ge=1,
        le=12,
        description="Max dynamic image tiles per image."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "images": ["test_data/satellite-image-t1.png"],
                "prompt": "<image>\nDescribe the terrain and land cover visible in this satellite observation.",
                "temperature": 0.0,
                "max_new_tokens": 128
            }
        }
    }


class ModelChatResponse(BaseModel):
    """Response returned by the EarthDial model direct inference."""
    success: bool = Field(default=True, description="Whether the inference succeeded.")
    prompt: str = Field(..., description="Prompt sent to the model.")
    response: str = Field(..., description="Generated text from the model.")
    num_images: int = Field(default=0, description="Number of images processed.")
    image_metadata: List[ImageResolutionInfo] = Field(default_factory=list, description="Metadata for each input image.")
    execution_time_seconds: float = Field(..., description="Latency in seconds.")
    device: str = Field(..., description="Execution device (e.g. cuda, cpu, mock).")
    model_name_or_path: str = Field(..., description="Model identifier or checkpoint path.")


class ModelInfoResponse(BaseModel):
    """Detailed information about the loaded EarthDial model."""
    model_path: str
    device: str
    model_loaded: bool
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_allocated_gb: Optional[float] = None
    image_size: int = 448
    architecture: str = "InternVL2 (Vision) + Phi-3 Mini (LLM)"
    parameters: str = "4.2 Billion"


# ============================================================================
# 3. System & Prompt Templates
# ============================================================================

class HealthResponse(BaseModel):
    """Health and runtime status response."""
    status: str
    model_loaded: bool
    model_path: str
    device: str
    cuda_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_allocated_gb: Optional[float] = None


class PromptTemplateItem(BaseModel):
    """A pre-configured prompt template for specific remote sensing tasks."""
    task_category: str
    name: str
    query: str
    suggested_tokens: str
    description: str


class PromptTemplatesResponse(BaseModel):
    """Collection of recommended prompt templates for EarthDial change detection."""
    templates: List[PromptTemplateItem]
