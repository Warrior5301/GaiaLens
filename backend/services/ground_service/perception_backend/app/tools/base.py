"""Base abstract tool interface for SAC GeoAI Agent.

Enforces:
- Generic typed input and output contracts (InputT, OutputT)
- Pydantic schema validation at entry and exit
- Standardized ToolExecutionResult packaging with latency metrics and error capture
- Conversion of perception results into traceable Evidence instances
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import time
from typing import Any, Generic, TypeVar
import uuid
from pydantic import ValidationError
from app.core.logging import logger
from app.schemas.evidence import Evidence
from app.schemas.tool_protocol import (
    ToolError,
    ToolErrorCode,
    ToolExecutionResult,
    ToolExecutionStatus,
    ToolInput,
    ToolOutput,
)

InputT = TypeVar("InputT", bound=ToolInput)
OutputT = TypeVar("OutputT", bound=ToolOutput)


class ToolExecutionException(Exception):
    """Exception carrying a structured ToolError to be propagated in ToolExecutionResult."""
    def __init__(self, error: ToolError):
        super().__init__(error.message)
        self.error = error


class BaseTool(ABC, Generic[InputT, OutputT]):
    """Abstract base class for all deterministic and perception tools."""

    name: str = "BaseTool"
    version: str = "1.0.0"
    description: str = "Base tool abstraction"
    input_schema_class: type[InputT]
    output_schema_class: type[OutputT]

    def validate_input(self, input_data: InputT | dict[str, Any]) -> InputT:
        """Validate raw dictionary or verify instance conforms to input_schema_class."""
        if isinstance(input_data, self.input_schema_class):
            return input_data
        if isinstance(input_data, dict):
            return self.input_schema_class.model_validate(input_data)
        raise TypeError(
            f"Expected {self.input_schema_class.__name__} or dict, got {type(input_data).__name__}"
        )

    def validate_output(self, output_data: OutputT | dict[str, Any]) -> OutputT:
        """Validate output conforms to output_schema_class."""
        if isinstance(output_data, self.output_schema_class):
            return output_data
        if isinstance(output_data, dict):
            return self.output_schema_class.model_validate(output_data)
        raise TypeError(
            f"Expected {self.output_schema_class.__name__} or dict, got {type(output_data).__name__}"
        )

    @abstractmethod
    def execute(self, input_data: InputT) -> OutputT:
        """Execute core tool logic on validated input. Must be overridden by subclasses."""
        pass

    def to_evidence(self, output: OutputT, asset_id: str) -> list[Evidence]:
        """Convert tool output into one or more verifiable Evidence units."""
        return []

    def run(
        self,
        input_data: InputT | dict[str, Any],
        call_id: str | None = None,
    ) -> ToolExecutionResult:
        """Standardized lifecycle execution wrapper: validates, times, captures errors, returns ToolExecutionResult."""
        cid = call_id or f"call_{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter()

        # Step 1: Input Validation
        try:
            validated_input = self.validate_input(input_data)
        except ValidationError as ve:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[{self.name}] Input validation failed: {ve}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[
                    ToolError(
                        code=ToolErrorCode.SCHEMA_VALIDATION_ERROR,
                        message=str(ve),
                        retryable=False,
                        details={"errors": ve.errors()},
                    )
                ],
                metrics={"latency_ms": latency_ms},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[{self.name}] Unexpected error validating input: {e}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[
                    ToolError(
                        code=ToolErrorCode.INVALID_INPUT,
                        message=str(e),
                        retryable=False,
                        details={},
                    )
                ],
                metrics={"latency_ms": latency_ms},
            )

        # Step 2: Tool Execution
        try:
            raw_output = self.execute(validated_input)
            validated_output = self.validate_output(raw_output)
            latency_ms = (time.perf_counter() - start_time) * 1000.0

            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.SUCCESS,
                data=validated_output.model_dump(),
                warnings=[],
                errors=[],
                metrics={"latency_ms": latency_ms},
            )
        except ToolExecutionException as tee:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[{self.name}] Tool returned structured error: {tee.error.code} - {tee.error.message}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[tee.error],
                metrics={"latency_ms": latency_ms},
            )
        except FileNotFoundError as fnf:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[{self.name}] Asset not found: {fnf}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[
                    ToolError(
                        code=ToolErrorCode.ASSET_NOT_FOUND,
                        message=str(fnf),
                        retryable=False,
                        details={},
                    )
                ],
                metrics={"latency_ms": latency_ms},
            )
        except ValidationError as ve:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.error(f"[{self.name}] Output validation failed: {ve}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[
                    ToolError(
                        code=ToolErrorCode.SCHEMA_VALIDATION_ERROR,
                        message=f"Output validation failed: {ve}",
                        retryable=False,
                        details={"errors": ve.errors()},
                    )
                ],
                metrics={"latency_ms": latency_ms},
            )
        except Exception as e:
            latency_ms = (time.perf_counter() - start_time) * 1000.0
            logger.exception(f"[{self.name}] Tool execution error: {e}")
            return ToolExecutionResult(
                call_id=cid,
                tool_name=self.name,
                tool_version=self.version,
                status=ToolExecutionStatus.FAILED,
                data=None,
                warnings=[],
                errors=[
                    ToolError(
                        code=ToolErrorCode.TOOL_INTERNAL_ERROR,
                        message=str(e),
                        retryable=False,
                        details={"exception_type": type(e).__name__},
                    )
                ],
                metrics={"latency_ms": latency_ms},
            )
