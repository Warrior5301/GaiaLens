# Contributing to SAC Shared Contracts

Thank you for contributing to the canonical integration contract repository for the **SAC GeoAI Agent**.

This repository is the single source of truth for all cross-team interfaces between:
- **Team A**: Image / VLM / Perception
- **Team B**: Geo / Spatial
- **Team C**: ROI / Multimodal / OCR

---

## 1. Golden Rules of Contract Evolution

1. **Never Redefine an Existing Shared Domain Type**: Common models such as `BBoxPx`, `OBB`, `AssetRef`, and `ArtifactRef` are immutable foundation types.
2. **Prefer Extending Existing Models**: Use inheritance or optional fields with strict validation rather than duplicating schemas.
3. **Semantic Versioning is Strictly Enforced**:
   - **Major (X.0.0)**: Breaking schema modifications (removing fields, tightening validations, renaming attributes).
   - **Minor (0.X.0)**: Additive backwards-compatible changes (new optional fields, new tool contract models).
   - **Patch (0.0.X)**: Bug fixes in schemas, docstring corrections, example enhancements.
4. **Regenerate JSON Schemas**: Whenever a Pydantic model is changed or added, run:
   ```bash
   python scripts/generate_schemas.py
   ```
   Committed generated schemas in `schemas/generated/` must match Pydantic source definitions exactly.
5. **Add Automated Tests**: Every contract change must include test coverage in `tests/`.
6. **Update Examples**: Update or add validated JSON payloads in `examples/`.
7. **Maintain the Changelog**: Record all additions, changes, and deprecations in `CHANGELOG.md`.
8. **Pull Requests Required**: Direct pushes to `main` are strictly forbidden. All contract changes must go through a reviewed Pull Request.

---

## 2. Local Development Workflow

```bash
# 1. Create a feature branch
git checkout -b feat/your-contract-feature

# 2. Implement your Pydantic model changes in src/sac_shared_contracts/

# 3. Regenerate schemas
python scripts/generate_schemas.py

# 4. Validate contracts
python scripts/validate_contracts.py

# 5. Run test suite
pytest

# 6. Commit and submit PR
git commit -m "feat(geo): add elevation profile contract"
git push -u origin feat/your-contract-feature
```
