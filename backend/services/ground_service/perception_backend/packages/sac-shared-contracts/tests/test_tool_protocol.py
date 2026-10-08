"""Tests for tool invocation and execution protocol contracts."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sac_shared_contracts.common import EvidenceRef
from sac_shared_contracts.tool_protocol import (
    ToolCall,
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolExecutionStatus,
)


class TestToolProtocol:
    def test_valid_tool_call(self):
        call = ToolCall(
            call_id="call_img_01",
            tool_name="Image_Analyzer",
            tool_version="0.1.0",
            arguments={"query": "Is there a runway in this image?"},
            depends_on=[],
        )
        assert call.call_id == "call_img_01"
        assert call.tool_name == "Image_Analyzer"
        assert call.arguments["query"] == "Is there a runway in this image?"

    def test_tool_call_forbids_extra_fields(self):
        with pytest.raises(ValidationError):
            ToolCall(
                call_id="c1",
                tool_name="test_tool",
                unexpected_flag="forbidden",
            )

    def test_valid_tool_error(self):
        err = ToolError(
            code=ToolErrorCode.GPU_OOM,
            message="CUDA out of memory while tiling raster",
            retryable=True,
            details={"attempted_tile_size": 2048, "available_vram_mb": 400},
        )
        assert err.code == ToolErrorCode.GPU_OOM
        assert err.retryable is True

    def test_standard_error_codes(self):
        codes = [
            "INVALID_INPUT",
            "ASSET_NOT_FOUND",
            "INVALID_ROI",
            "UNSUPPORTED_MODALITY",
            "CRS_MISMATCH",
            "REGISTRATION_FAILED",
            "MODEL_TIMEOUT",
            "GPU_OOM",
            "LOW_CONFIDENCE",
            "NO_DETECTIONS",
            "SCHEMA_VALIDATION_ERROR",
            "TOOL_INTERNAL_ERROR",
        ]
        for c in codes:
            err = ToolError(code=c, message="test error")
            assert err.code == c

    def test_tool_execution_result_success(self):
        ev = EvidenceRef(
            evidence_id="ev_01",
            source_tool="Image_Analyzer",
            confidence=0.95,
        )
        result = ToolExecutionResult(
            call_id="call_img_01",
            tool_name="Image_Analyzer",
            tool_version="0.1.0",
            status=ToolExecutionStatus.SUCCESS,
            data={"answer": "Yes, a paved commercial runway is clearly visible."},
            evidence_refs=[ev],
            metrics={"latency_ms": 320.0},
        )
        assert result.status == ToolExecutionStatus.SUCCESS
        assert len(result.evidence_refs) == 1
        assert result.errors == []

    def test_tool_execution_result_failed_with_errors(self):
        err = ToolError(
            code=ToolErrorCode.MODEL_TIMEOUT,
            message="Model inference exceeded 30s threshold",
            retryable=True,
        )
        result = ToolExecutionResult(
            call_id="call_img_02",
            tool_name="Image_Analyzer",
            tool_version="0.1.0",
            status=ToolExecutionStatus.FAILED,
            errors=[err],
        )
        assert result.status == ToolExecutionStatus.FAILED
        assert len(result.errors) == 1
        assert result.errors[0].retryable is True
