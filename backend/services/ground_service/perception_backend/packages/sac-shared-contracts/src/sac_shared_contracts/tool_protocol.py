"""Standardized tool invocation and execution protocol contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional
from pydantic import Field
from sac_shared_contracts.common import EvidenceRef, StrictModel


class ToolInput(StrictModel):
    """Marker base model for all tool input contracts."""
    pass


class ToolOutput(StrictModel):
    """Marker base model for all tool output contracts."""
    pass


class ToolExecutionStatus(str, Enum):
    """Lifecycle completion status of a tool invocation."""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"


class ToolErrorCode(str, Enum):
    """Standardized machine-readable error codes across all SAC tools."""
    INVALID_INPUT = "INVALID_INPUT"
    ASSET_NOT_FOUND = "ASSET_NOT_FOUND"
    INVALID_ROI = "INVALID_ROI"
    UNSUPPORTED_MODALITY = "UNSUPPORTED_MODALITY"
    CRS_MISMATCH = "CRS_MISMATCH"
    REGISTRATION_FAILED = "REGISTRATION_FAILED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    GPU_OOM = "GPU_OOM"
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    NO_DETECTIONS = "NO_DETECTIONS"
    SCHEMA_VALIDATION_ERROR = "SCHEMA_VALIDATION_ERROR"
    TOOL_INTERNAL_ERROR = "TOOL_INTERNAL_ERROR"


class ToolError(StrictModel):
    """Structured error descriptor returned during tool failure."""
    code: ToolErrorCode | str = Field(description="Standardized machine-readable error code")
    message: str = Field(description="Human-readable explanation of error condition")
    retryable: bool = Field(
        default=False, description="Whether transient condition permits an automatic retry"
    )
    details: dict[str, Any] = Field(
        default_factory=dict, description="Diagnostic payload, stack traces, or validation details"
    )


class ToolCall(StrictModel):
    """Standardized tool invocation envelope submitted by the orchestrator/agent."""
    call_id: str = Field(description="Unique invocation identifier")
    tool_name: str = Field(description="Canonical identifier of registered tool")
    tool_version: str = Field(default="1.0.0", description="Semantic version of target tool contract")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters passed to the tool"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="IDs of prerequisite tool calls that must complete first"
    )


class ToolExecutionResult(StrictModel):
    """Standardized tool response envelope returned by tool runtimes."""
    call_id: str = Field(description="Invocation identifier corresponding to ToolCall")
    tool_name: str = Field(description="Name of executed tool")
    tool_version: str = Field(description="Version of executed tool")
    status: ToolExecutionStatus = Field(description="Execution outcome status")
    data: Optional[dict[str, Any]] = Field(
        default=None, description="Serialized ToolOutput payload on success or partial success"
    )
    evidence_refs: list[EvidenceRef] = Field(
        default_factory=list, description="References to factual evidence generated during execution"
    )
    warnings: list[str] = Field(
        default_factory=list, description="Non-fatal warnings or caveats emitted during run"
    )
    errors: list[ToolError] = Field(
        default_factory=list, description="Structured error records if failures occurred"
    )
    metrics: dict[str, Any] = Field(
        default_factory=dict, description="Execution performance metrics (e.g. latency_ms, tiles_processed)"
    )
