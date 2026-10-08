"""Tool and Model Registry implementations.

Provides:
- ToolDefinition and ModelDefinition schema representations
- Loading and self-validating configuration from YAML
- Capability-to-tool resolution
- Dynamic instantiation with dependency injection
"""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Optional
import yaml
from pydantic import Field
from app.core.logging import logger
from app.schemas.common import StrictModel
from app.tools.base import BaseTool


class ToolImplementation(StrictModel):
    module: str = Field(description="Python import path to the tool module")
    tool_class: str = Field(alias="class", description="Name of the BaseTool subclass")


class ToolDefinition(StrictModel):
    name: str = Field(description="Unique tool identifier")
    version: str = Field(description="Semantic version string")
    category: str = Field(description="Tool functional category (e.g. perception, geometry, analysis)")
    description: str = Field(description="Human-readable description of tool behavior")
    capabilities: list[str] = Field(description="List of capability tags supported by tool")
    input_schema: str = Field(description="Relative reference to JSON schema for input")
    output_schema: str = Field(description="Relative reference to JSON schema for output")
    implementation: ToolImplementation = Field(description="Module and class pointers")
    cost_class: str = Field(default="medium", description="Computational cost category: low, medium, high")
    deterministic: bool = Field(default=False, description="Whether tool execution is strictly deterministic")
    parallelizable: bool = Field(default=True, description="Whether tool can run concurrently with other tools")
    supported_modalities: list[str] = Field(
        default_factory=lambda: ["optical"], description="Supported sensor modalities"
    )


class ModelWeights(StrictModel):
    uri: str = Field(description="Path or URL to model checkpoint weights")
    sha256: Optional[str] = Field(default=None, description="Optional SHA256 checksum of weights")


class ModelDefinition(StrictModel):
    model_id: str = Field(description="Unique model identifier")
    task: str = Field(description="Target task type (e.g. captioning, detection, grounding)")
    framework: str = Field(description="Deep learning framework: pytorch, onnx, etc.")
    architecture: str = Field(description="Underlying neural network architecture")
    weights: ModelWeights = Field(description="Checkpoint weight configuration")
    precision: str = Field(default="fp16", description="Floating point precision: fp32, fp16, bf16, int8")
    device: Optional[str] = Field(default=None, description="Target device override (cuda, cpu, mps)")
    hyperparameters: dict[str, Any] = Field(
        default_factory=dict, description="Model-specific inference hyperparameters"
    )


class ToolRegistry:
    """Central registry and factory for all perception and analysis tools."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.tools: dict[str, ToolDefinition] = {}
        self.capability_index: dict[str, list[str]] = {}
        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: str | Path) -> None:
        """Load and self-validate tool definitions from YAML config."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Tool registry configuration not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not raw_data or "tools" not in raw_data:
            raise ValueError(f"Invalid tool registry format in {path}: 'tools' root key missing")

        tools_data = raw_data["tools"]
        for tool_name, tool_dict in tools_data.items():
            if "name" not in tool_dict:
                tool_dict["name"] = tool_name
            tool_def = ToolDefinition.model_validate(tool_dict)
            self.register_tool(tool_def)

        logger.info(f"Loaded {len(self.tools)} tool definitions into ToolRegistry")

    def register_tool(self, tool_def: ToolDefinition) -> None:
        """Register a tool definition and index its capabilities."""
        if tool_def.name in self.tools:
            raise ValueError(f"Duplicate tool name registered: {tool_def.name}")

        self.tools[tool_def.name] = tool_def
        for cap in tool_def.capabilities:
            if cap not in self.capability_index:
                self.capability_index[cap] = []
            self.capability_index[cap].append(tool_def.name)

    def get_tool_definition(self, name: str) -> ToolDefinition:
        """Retrieve tool metadata by canonical name."""
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' is not registered in ToolRegistry")
        return self.tools[name]

    def list_tools(self) -> list[ToolDefinition]:
        """Return all registered tools."""
        return list(self.tools.values())

    def list_capabilities(self) -> list[str]:
        """List all indexed tool capabilities."""
        return list(self.capability_index.keys())

    def resolve_tool_for_capability(self, capability: str) -> list[str]:
        """Find tool names matching a given capability requirement."""
        return self.capability_index.get(capability, [])

    def instantiate_tool(self, name: str, **kwargs: Any) -> BaseTool:
        """Instantiate a tool class dynamically by looking up its definition."""
        tool_def = self.get_tool_definition(name)
        mod_name = tool_def.implementation.module
        cls_name = tool_def.implementation.tool_class

        try:
            mod = importlib.import_module(mod_name)
        except ImportError as e:
            raise ImportError(f"Could not import module '{mod_name}' for tool '{name}': {e}") from e

        if not hasattr(mod, cls_name):
            raise AttributeError(f"Module '{mod_name}' does not contain tool class '{cls_name}'")

        tool_cls = getattr(mod, cls_name)
        return tool_cls(**kwargs)


class ModelRegistry:
    """Registry mapping model IDs to architectures and checkpoint specifications."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.models: dict[str, ModelDefinition] = {}
        if config_path:
            self.load_from_yaml(config_path)

    def load_from_yaml(self, config_path: str | Path) -> None:
        """Load and validate model definitions from YAML config."""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Model registry configuration not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            raw_data = yaml.safe_load(f)

        if not raw_data or "models" not in raw_data:
            raise ValueError(f"Invalid model registry format in {path}: 'models' root key missing")

        for model_id, model_dict in raw_data["models"].items():
            if "model_id" not in model_dict:
                model_dict["model_id"] = model_id
            model_def = ModelDefinition.model_validate(model_dict)
            self.register_model(model_def)

        logger.info(f"Loaded {len(self.models)} model definitions into ModelRegistry")

    def register_model(self, model_def: ModelDefinition) -> None:
        """Register model definition."""
        if model_def.model_id in self.models:
            raise ValueError(f"Duplicate model_id registered: {model_def.model_id}")
        self.models[model_def.model_id] = model_def

    def get_model_definition(self, model_id: str) -> ModelDefinition:
        """Retrieve model metadata by ID."""
        if model_id not in self.models:
            raise KeyError(f"Model '{model_id}' is not registered in ModelRegistry")
        return self.models[model_id]

    def list_models(self) -> list[ModelDefinition]:
        return list(self.models.values())
