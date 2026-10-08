"""Configuration validator script for SAC GeoAI Agent.

Validates:
1. Tool registry YAML against ToolDefinition schema
2. Model registry YAML against ModelDefinition schema
3. Agent policy and runtime configuration
4. Checks that all referenced schema files exist on disk
5. Checks that all tool classes can be dynamically resolved and imported
"""

from __future__ import annotations

from pathlib import Path
import sys
import yaml

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.models.registry import ModelRegistry, ToolRegistry


def validate_all_configs(config_dir: str | Path = "config") -> int:
    cfg_path = Path(config_dir)
    print(f"Validating configurations in '{cfg_path.resolve()}'...")
    errors: list[str] = []

    # 1. Validate Tool Registry
    tool_reg_path = cfg_path / "tool_registry.yaml"
    if not tool_reg_path.exists():
        errors.append(f"Missing {tool_reg_path}")
    else:
        try:
            tool_registry = ToolRegistry(tool_reg_path)
            print(f"  [OK] tool_registry.yaml: {len(tool_registry.tools)} tools registered")

            for name, tool_def in tool_registry.tools.items():
                # Check schema files
                in_schema = REPO_ROOT / tool_def.input_schema
                out_schema = REPO_ROOT / tool_def.output_schema
                if not in_schema.exists():
                    errors.append(f"Tool '{name}' references missing input schema: {tool_def.input_schema}")
                if not out_schema.exists():
                    errors.append(f"Tool '{name}' references missing output schema: {tool_def.output_schema}")

                # Check dynamic class instantiation
                try:
                    tool_instance = tool_registry.instantiate_tool(name)
                    print(f"    - Tool '{name}' ({tool_instance.__class__.__name__}) instantiated successfully")
                except Exception as ex:
                    errors.append(f"Failed to instantiate tool '{name}': {ex}")

        except Exception as e:
            errors.append(f"Error validating tool_registry.yaml: {e}")

    # 2. Validate Model Registry
    model_reg_path = cfg_path / "model_registry.yaml"
    if not model_reg_path.exists():
        errors.append(f"Missing {model_reg_path}")
    else:
        try:
            model_registry = ModelRegistry(model_reg_path)
            print(f"  [OK] model_registry.yaml: {len(model_registry.models)} models registered")
            for m in model_registry.list_models():
                print(f"    - Model '{m.model_id}': task={m.task}, arch={m.architecture}")
        except Exception as e:
            errors.append(f"Error validating model_registry.yaml: {e}")

    # 3. Validate Agent Policy
    policy_path = cfg_path / "agent_policy.yaml"
    if not policy_path.exists():
        errors.append(f"Missing {policy_path}")
    else:
        try:
            with open(policy_path, "r", encoding="utf-8") as f:
                policy = yaml.safe_load(f)
            if "routing" not in policy:
                errors.append("agent_policy.yaml missing 'routing' block")
            else:
                print(f"  [OK] agent_policy.yaml: {len(policy['routing'])} routing tasks defined")
        except Exception as e:
            errors.append(f"Error validating agent_policy.yaml: {e}")

    # 4. Validate Runtime Config
    runtime_path = cfg_path / "runtime.yaml"
    if not runtime_path.exists():
        errors.append(f"Missing {runtime_path}")
    else:
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                rt = yaml.safe_load(f)
            if "runtime" not in rt:
                errors.append("runtime.yaml missing 'runtime' block")
            else:
                print("  [OK] runtime.yaml: runtime parameters valid")
        except Exception as e:
            errors.append(f"Error validating runtime.yaml: {e}")

    if errors:
        print("\nConfiguration validation failed with errors:", file=sys.stderr)
        for err in errors:
            print(f"  [FAIL] {err}", file=sys.stderr)
        return 1

    print("\nAll configuration files are fully valid and operational.")
    return 0


if __name__ == "__main__":
    sys.exit(validate_all_configs())
