# SAC Shared Contracts (`sac-shared-contracts`)

[![Tests and Contract Validation](https://github.com/ansh369-hub/sac-shared-contracts/actions/workflows/tests.yml/badge.svg)](https://github.com/ansh369-hub/sac-shared-contracts/actions/workflows/tests.yml)
![Python Version](https://img.shields.io/badge/python-%3E%3D3.11-blue.svg)
![Pydantic](https://img.shields.io/badge/pydantic-v2-green.svg)
![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)

Canonical shared schemas, domain models, tool protocol, and integration contracts for the **SAC GeoAI Agent** ecosystem.

This repository serves as the **single source of truth** for data contracts, geometry conventions, tool invocation payloads, and evidence models shared across all workstreams. It is a lightweight integration dependency and contains **NO model weights, satellite imagery, credentials, or large datasets**.

---

## 1. Architectural Philosophy

1. **Pydantic Models are the Source of Truth**: All domain and tool models are defined in Python using Pydantic v2.
2. **Deterministic JSON Schema Generation**: JSON Schemas (Draft 2020-12) are compiled directly from Pydantic models. Never hand-edit JSON schemas.
3. **YAML is Configuration Only**: Tool registries and metadata configs use YAML validated against our JSON schemas.
4. **Lightweight Integration**: Heavy image arrays and model state are decoupled from agent messages via `AssetRef` and `ArtifactRef`.

```
        +-----------------------------------+
        |  Pydantic Models (Source of Truth)|
        +-----------------------------------+
                          |
                          v
        +-----------------------------------+
        | JSON Schema Gen (Draft 2020-12)   |
        +-----------------------------------+
                          |
                          v
        +-----------------------------------+
        |    sac-shared-contracts Package   |
        +-----------------------------------+
              /           |           \
             v            v            v
        Team A          Team B       Team C
     (Perception)    (Geo/Spatial) (ROI/Multimodal)
```

---

## 2. Frozen Coordinate Conventions

> **CRITICAL ARCHITECTURAL CONTRACT**: These coordinate and orientation definitions are frozen project-wide. No team or tool runtime may deviate from or re-interpret these conventions.

### Pixel Space
- **Origin**: `(0, 0)` is at the **top-left** corner of the image raster.
- **X-axis**: Extends horizontally to the **right** (`x >= 0`).
- **Y-axis**: Extends vertically **downwards** (`y >= 0`).

### Axis-Aligned Bounding Box (`BBoxPx`)
Defined by four bounding coordinates: `[x_min, y_min, x_max, y_max]`.
Strict invariant enforced:
$$\text{x\_max} \ge \text{x\_min} \quad \text{and} \quad \text{y\_max} \ge \text{y\_min}$$

### Oriented Bounding Box (`OBB`)
Defined by center coordinates, dimensions, and rotation:
- `cx`, `cy`: Center coordinates in pixel space.
- `width`: Dimension along the major box axis ($\ge 0$).
- `height`: Dimension along the minor box axis ($\ge 0$).
- `angle_deg`: Orientation angle in degrees.
- **Angle Convention**: `ccw_from_image_x` (Counter-Clockwise from positive horizontal image X-axis).

---

## 3. Asset and Artifact Separation

To prevent agent context bloat and out-of-memory errors:

- **`AssetRef` (Input Data)**: Represents external input satellite/aerial imagery. Contains metadata (`asset_id`, `uri`, `modality`, `width_px`, `height_px`, `gsd_m`, `crs`, `acquisition_time`, `bands`). Actual image tensors are never stored in `AssetRef`.
- **`ArtifactRef` (Derived Data)**: Represents files produced by tools (e.g. segmentation masks, crops, shapefiles, fusion layers). Contains `artifact_id`, `artifact_type`, `uri`, `source_asset_ids`, and `metadata`.
- **Supported Modalites**: `optical`, `sar`, `infrared`, `thermal`, `multispectral`, `hyperspectral`, `panchromatic`.

---

## 4. Team Ownership & Responsibilities

| Workstream | Tools Owned | Focus Area |
| :--- | :--- | :--- |
| **Team A — Perception** | `Image_Analyzer`, `Object_Detector`, `OBB_Detector`, `Segmentation_Grounding` | Remote-sensing VLMs, horizontal & oriented object detection, promptable segmentation |
| **Team B — Geo / Spatial** | `Geo_Measurement`, `Spatial_Relationship`, `Coordinate_Transformer` | GSD scaling, surface area/perimeter computation, topological & directional relations |
| **Team C — ROI / Multimodal**| `ROI_Localizer`, `SAR_Optical_Fuser`, `OCR_Extractor` | Conversational ROI proposal, coregistration/fusion (SAR + optical), scene OCR |

Modifications to `common.py`, `tool_protocol.py`, or cross-team interfaces require review from all affected teams via PR.

---

## 5. Installation

```bash
# Direct pip install from Git repository (Private repo authentication required)
pip install git+https://github.com/ansh369-hub/sac-shared-contracts.git

# Or install for local development with dev tools:
git clone https://github.com/ansh369-hub/sac-shared-contracts.git
cd sac-shared-contracts
pip install -e .[dev]
```

---

## 6. Quickstart Import Examples

```python
from sac_shared_contracts.common import AssetRef, BBoxPx, OBB, Modality, ArtifactRef
from sac_shared_contracts.tool_protocol import ToolCall, ToolExecutionResult
from sac_shared_contracts.geo_spatial import GeoMeasurementInput, GeometryRef
from sac_shared_contracts.roi import ROI, ROILocalizerInput
from sac_shared_contracts.multimodal import FusionInput
from sac_shared_contracts.ocr import OCRInput

# 1. Define an input asset reference
asset = AssetRef(
    asset_id="scene_001",
    uri="s3://satellite-data/optical/scene_001.tif",
    modality=Modality.OPTICAL,
    width_px=4096,
    height_px=4096,
    gsd_m=0.3,
    crs="EPSG:3857"
)

# 2. Define an Oriented Bounding Box
obb = OBB(
    cx=1200.0,
    cy=850.0,
    width=65.0,
    height=22.0,
    angle_deg=35.0,
    angle_convention="ccw_from_image_x"
)
```

---

## 7. Cross-Team Integration Workflows

### Pipeline 1: Conversational Query -> ROI -> Object Detector
1. Orchestrator calls `ROI_Localizer` with query `"harbor container cranes"`.
2. `ROI_Localizer` returns `ROILocalizerOutput` with `rois: list[ROI]`.
3. Orchestrator routes each `ROI.bbox_px` to `ObjectDetectorInput.roi`.

```python
from sac_shared_contracts.roi import ROILocalizerOutput
from sac_shared_contracts.perception import ObjectDetectorInput

# Team C produces ROI
roi_output: ROILocalizerOutput = ...
primary_roi = roi_output.rois[0]

# Team A consumes ROI
detector_input = ObjectDetectorInput(
    asset=sample_asset,
    roi=primary_roi.bbox_px,
    target_classes=["container_crane"],
    confidence_threshold=0.5
)
```

### Pipeline 2: Segmentation Mask -> Geo-Measurement
1. `Segmentation_Grounding` produces a binary mask saved as an `ArtifactRef(artifact_type="segmentation_mask")`.
2. `Geo_Measurement` receives `GeoMeasurementInput` where `geometry = GeometryRef(geometry_type="artifact", artifact_id=mask_artifact.artifact_id)`.
3. `Geo_Measurement` computes ground area and perimeter in metric units.

```python
from sac_shared_contracts.geo_spatial import GeoMeasurementInput, GeometryRef

measurement_input = GeoMeasurementInput(
    asset=sample_asset,
    geometry=GeometryRef(
        geometry_type="artifact",
        artifact_id="mask_storage_tank_01"
    ),
    measurement_types=["area", "perimeter"]
)
```

---

## 8. Development & Verification Workflow

```bash
# 1. Run all unit tests
python -m pytest -v

# 2. Generate updated JSON Schemas
python scripts/generate_schemas.py

# 3. Validate contracts, test models, and verify JSON examples
python scripts/validate_contracts.py
```

---

## 9. Versioning Policy

This package follows [Semantic Versioning 2.0.0](https://semver.org/):
- **PATCH** (`0.1.x`): Backwards-compatible bug fixes, documentation, or validator adjustments.
- **MINOR** (`0.x.0`): Additive schema extensions (new optional fields, new tool models).
- **MAJOR** (`x.0.0`): Breaking contract changes (field renames, removed fields, coordinate convention modifications).
