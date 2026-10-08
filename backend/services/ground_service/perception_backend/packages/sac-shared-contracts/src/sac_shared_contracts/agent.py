"""Framework-independent agent planning, claim verification, and response contracts."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import Field
from sac_shared_contracts.common import StrictModel


class PlanStep(StrictModel):
    """Single orchestrated tool invocation step within an execution plan."""
    step_id: str = Field(description="Unique step identifier (e.g. 'step_1')")
    tool_name: str = Field(description="Canonical identifier of tool to invoke")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Input parameters passed to the tool contract"
    )
    depends_on: list[str] = Field(
        default_factory=list, description="List of step IDs that must complete before this step runs"
    )
    description: Optional[str] = Field(
        default=None, description="Human-readable rationale for this execution step"
    )


class AgentPlan(StrictModel):
    """Decomposed multi-step execution plan produced by the planner."""
    request_id: str = Field(description="Original user request identifier")
    steps: list[PlanStep] = Field(
        default_factory=list, description="Ordered or dependency-directed steps to execute"
    )
    reasoning: Optional[str] = Field(
        default=None, description="High-level plan decomposition rationale"
    )


class Claim(StrictModel):
    """Atomic verifiable factual statement made in the final synthesized answer."""
    claim_id: str = Field(description="Unique claim identifier (e.g. 'c1')")
    text: str = Field(description="Textual representation of the claim")
    claim_type: str = Field(
        description="Type classification: count, existence, localization, measurement, spatial_relation, attribute"
    )
    value: Any = Field(
        default=None, description="Extracted numerical or categorical value for deterministic verification"
    )
    unit: Optional[str] = Field(
        default=None, description="Unit of measurement if applicable (e.g. 'm2', 'km', 'pixels')"
    )
    supported_by: list[str] = Field(
        default_factory=list, description="List of evidence IDs justifying this claim"
    )


class FinalResponse(StrictModel):
    """Structured final response payload with verifiable evidence links."""
    answer: str = Field(description="Natural language response synthesized for the user")
    claims: list[Claim] = Field(
        default_factory=list, description="Traceable atomic claims embedded in the answer"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall answer confidence based on supporting evidence"
    )
    limitations: list[str] = Field(
        default_factory=list, description="Explicit caveats, model uncertainties, or sensor assumptions"
    )
