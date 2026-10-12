# Repository consolidation

The former `repo/aiml_04` repository is now `isro-vlm`, with its Git history preserved. The outer `backend` and `frontend` were merged into it at the user's request. Outer files are active where versions differed; files found only in the original repository remain active too.

## Findings

- The outer backend had 334 byte-identical files; the outer frontend had 17.
- Of 58 files whose bytes differed, 47 differed only in line endings and 11 had substantive differences.
- The outer frontend added `captions`, `explore-earthmind`, `query-orchestrator`, and `terrascope` API routes; the backend added `local_query_runner.py`.
- Other outer-only files included local environment settings, an input archive, and temporary images. Generated Next.js/TypeScript output was not treated as source.
- `workspace` contained no files. `trash` held original ZIP distributions, now kept in local recovery instead of discarded without review.
- The duplicate `next.config.js` was removed after its ArcGIS transpilation, image and Turbopack settings were consolidated into `next.config.ts`. The obsolete `experimental.turbo` setting was not carried over.
- Service directories with similar names were retained: their names alone do not establish that they are redundant.

## Recovery

`local-recovery/` is intentionally ignored by Git and remains on this machine. It can include private environment settings, so do not publish it wholesale.

| Artifact | Contents |
|---|---|
| `source-versions.zip` | Both versions of conflicting files (including line-ending differences), outer-only files, the original root README, and the outer frontend's Git metadata |
| `merge-manifest.json` | Paths, conflict classification, and SHA-256 hashes for both originals |
| `original-archives/` | Original `app.zip`, `frontend.zip`, and `services.zip` from `trash` |

Inside the source ZIP, `outer/` holds the former outer files and `repo-aiml_04/` holds repository originals. ZIP integrity and each manifest file's SHA-256 were checked before merging. Substantive conflicts were:

```text
backend/app/intelli/common.py
backend/app/intelli/grounding/ground.py
backend/app/intelli/orchestrator/orch.py
backend/app/intelli/orchestrator/tools/image_context.py
frontend/package.json
frontend/src/app/api/chat/route.ts
frontend/src/app/api/detect/route.ts
frontend/src/app/files/[name]/route.ts
frontend/src/components/ExploreEarth/ArcGISViewer.tsx
frontend/src/components/HomePage/Navbar.tsx
frontend/src/components/shared/AnalysisPage.tsx
```

To recover, open the ZIP and extract the desired original into a separate comparison directory. Compare it with the active file before replacing anything. Existing Git history also preserves previously committed repository versions; no consolidation commit was created automatically.

The retained source is not a blind deletion of one project version: both versions are recoverable, the extra routes and runner are active, and original distributions are retained locally. Generated `.next` output, Python bytecode and old TypeScript build metadata can be recreated. Installed frontend dependencies were moved for local validation.

## Validation after consolidation

- All 425 original non-generated outer files were verified against the merged file or the hash-checked recovery ZIP before deletion.
- Git HEAD remains `5bf4942` (`Update README.md`); history was retained.
- All 249 Python source files parsed successfully with Python's AST parser. This does not import GPU dependencies or exercise inference.
- `npm run build` passed, including TypeScript and page generation. Turbopack reported one warning: the dynamic Python subprocess invocation in `/api/query-orchestrator` causes broad project tracing during packaging.
- `npm run lint` completed with 54 errors and 25 warnings in the retained frontend source. The full report is local at `local-recovery/eslint-results.json`; lint is not a passing check.
- README file links, the final single-project layout, retained local environment settings, and the single Next.js config were checked.
- Live model endpoints, GPU inference and authenticated Earth Engine workflows were not exercised.
