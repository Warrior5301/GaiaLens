"""Perception and tool-specific I/O schemas for SAC GeoAI Agent.

Canonical definitions imported directly from sac_shared_contracts.
"""

from __future__ import annotations

from sac_shared_contracts.perception import (
    GenerationConfig,
    GroundingResult,
    ImageAnalyzerInput,
    ImageAnalyzerOutput,
    ImageTaskType,
    NMSConfig,
    OBBDetectorInput,
    OBBDetectorOutput,
    ObjectDetectorInput,
    ObjectDetectorOutput,
    RotatedNMSConfig,
    SegmentationGroundingInput,
    SegmentationGroundingOutput,
    TilingConfig,
)

__all__ = [
    "GenerationConfig",
    "NMSConfig",
    "RotatedNMSConfig",
    "TilingConfig",
    "ImageTaskType",
    "ImageAnalyzerInput",
    "ImageAnalyzerOutput",
    "ObjectDetectorInput",
    "ObjectDetectorOutput",
    "OBBDetectorInput",
    "OBBDetectorOutput",
    "GroundingResult",
    "SegmentationGroundingInput",
    "SegmentationGroundingOutput",
]
