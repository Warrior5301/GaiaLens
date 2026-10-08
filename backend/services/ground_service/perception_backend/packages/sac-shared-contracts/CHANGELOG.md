# Changelog

All notable changes to the `sac-shared-contracts` package will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-10-01

### Added
- Added backward-compatible `source_asset_id` pre-validation to `ArtifactRef` (auto-maps singular `source_asset_id` into canonical list `source_asset_ids`).
- Extended `ImageAnalyzerInput.roi` and `ObjectDetectorInput.roi` to support both `BBoxPx` and `ROI` instances directly.

### Changed
- Aligned `ToolCall.tool_version` default value to `"1.0.0"` to match registered tool versioning.

## [0.1.0] - 2026-10-01

### Added
- Initial canonical shared domain models: `StrictModel`, `Modality`, `CoordinateFrame`, `PointPx`, `BBoxPx`, `GeoPoint`, `OBB`, `Confidence`, `AssetRef`, `ArtifactRef`, `EvidenceRef`, `Detection`.
- Standard tool protocol contracts: `ToolInput`, `ToolOutput`, `ToolCall`, `ToolExecutionResult`, `ToolError`, and `ToolErrorCode`.
- Team A Perception contracts: `ImageAnalyzerInput/Output`, `ObjectDetectorInput/Output`, `OBBDetectorInput/Output`, `SegmentationGroundingInput/Output`.
- Team B Geo/Spatial contracts: `GeometryRef` (discriminated), `MeasurementValue`, `GeoMeasurementInput/Output`, `SpatialRelationshipInput/Output`.
- Team C ROI/Multimodal/OCR contracts: `ROI`, `ROILocalizerInput/Output`, `ModalityAsset`, `FusionInput`, `AlignmentReport`, `FusionOutput`, `OCRTextRegion`, `OCRInput/Output`.
- Evidence tracing schema: `Evidence`.
- Agent orchestration schemas: `PlanStep`, `AgentPlan`, `Claim`, `FinalResponse`.
- Automated JSON Schema generation script (`scripts/generate_schemas.py`) targeting Draft 2020-12.
- Contract validation script (`scripts/validate_contracts.py`).
- Continuous Integration workflow (`.github/workflows/tests.yml`).
