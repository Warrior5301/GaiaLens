"""
EarthDial Temporal Comparison Tool & Model Package
"""

from model import EarthDialModel
from tool import TemporalComparisonTool
from schemas import (
    HealthResponse,
    ImageResolutionInfo,
    ModelChatRequest,
    ModelChatResponse,
    ModelInfoResponse,
    PromptStyle,
    PromptTemplateItem,
    PromptTemplatesResponse,
    TemporalComparisonRequest,
    TemporalComparisonResponse,
)

# Backward-compatible alias
EarthDialTemporalComparator = TemporalComparisonTool

__all__ = [
    "EarthDialModel",
    "TemporalComparisonTool",
    "EarthDialTemporalComparator",
    "TemporalComparisonRequest",
    "TemporalComparisonResponse",
    "ModelChatRequest",
    "ModelChatResponse",
    "ModelInfoResponse",
    "HealthResponse",
    "PromptStyle",
    "PromptTemplateItem",
    "PromptTemplatesResponse",
    "ImageResolutionInfo",
]
