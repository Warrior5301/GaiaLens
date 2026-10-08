# SAC Shared Contracts — Generated JSON Schemas

This directory contains the canonical JSON Schemas automatically generated from the Pydantic domain models in `src/sac_shared_contracts/`.

> **CRITICAL RULE**: Do **NOT** hand-edit any file in `schemas/generated/`.
> The Python Pydantic models are the single source of truth. JSON Schemas are generated deterministically.

## Flow

```
Pydantic Models (src/sac_shared_contracts/)
               │
               ▼
   scripts/generate_schemas.py
               │
               ▼
Generated Schemas (schemas/generated/*.schema.json)
```

## Schema Standard

All generated schemas adhere to:
- **Specification**: [JSON Schema Draft 2020-12](https://json-schema.org/draft/2020-12/schema)
- **$schema**: `https://json-schema.org/draft/2020-12/schema`
- **$id**: Unique URI identifying the canonical schema contract
- **x-schema-version**: Current semantic contract release (e.g. `0.1.0`)

## Generation Command

To regenerate all schemas after modifying a Pydantic model:

```bash
python scripts/generate_schemas.py
```

## Validation Command

To verify schema generation and validate against examples and test payloads:

```bash
python scripts/validate_contracts.py
```
