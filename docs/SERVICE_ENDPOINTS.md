# Which service each agent tool calls

Both web chats send requests to Next.js `/api/chat`, then FastAPI `/chat`. FastAPI refines the question with Moondream and starts the Qwen orchestrator. The orchestrator chooses specialist tools; a tool makes an HTTP request only when the agent selects it. These are **HTTP endpoints**, so the model service can run locally or on Modal. A Modal deployment normally returns a hosted HTTPS URL, not a localhost address.

Set the variables **in the FastAPI process environment before starting it**. `backend/app/intelli/common.py` reads them at import time. The backend does not automatically load a `.env` file. Use the service's base URL without `/docs`; the tool appends the route shown below. `YOLOVR_URL` may include `/predict` if that is the URL you received.

| Caller | Environment variable | Route appended | Service source | Deployment source in this repo |
|---|---|---|---|---|
| Qwen agents and optional LoRA loader | `QWEN_URL` | OpenAI-compatible `/v1` API; `/load_lora_adapter` for adapters | `backend/app/modal_vllm/` | `modal_vllm/vllm_server.py` |
| Query refinement, called before every chat request | `MOONDREAM_URL` | `POST /run`; `POST /load` on a 409 | `services/moondream/moondream/` | `modal_app.py` |
| Grounding `object_detection` | `GROUND_URL` | `POST /tools/object_detector/run`; `/load` on a 409 | `services/ground_service/Ground_service/perception_backend/` | `deployment/modal/perception_tools_modal.py` |
| Grounding `yolo_object_detection` | `YOLOVR_URL` (also accepts `YOLOVR_API_URL`) | `POST /predict` | `services/yolo_serve/yolovr/` | `modal_app.py` |
| Grounding `segmentation` and `open_language_grounding` | `SAM3_URL` | `POST /run` | `services/poly_service/SAM3_endpoint/` | No Modal wrapper in this checkout; FastAPI app is at `app/main.py` |
| Orchestrator `ocr_tool` | `OCR_URL` | `POST /run` | `services/ocr_service/ocr_serve/` | Dockerfile, no Modal wrapper in this checkout |
| Orchestrator `image_analyzer` GSD lookup | `GSD_URL` | `POST /run` | `services/gsd_service/gsd_dataset/` | `modal_app.py` |
| Orchestrator `image_enhancer` | `ENHANCER_URL` | `POST /load`, `/run`, `/unload` | `services/image_enhancer/SlimDiffSR_API/` | Dockerfile, no Modal wrapper in this checkout |
| Orchestrator `image_context` | `CONTEXT_URL` | `POST /location-explorer` | `services/web_searching/websearch/` | `modal_fastapi_endpoint.py` |
| SAR/multispectral specialist | `EARTHMIND_URL` | `POST /run` | `services/earthmind/api_earthmind/` | `modal_app.py` |
| Temporal specialist, multi-image | `EARTHDIAL_URL` | `POST /compare` | `services/earthdial/temporal_comparison_tool/` | `modal_app.py` |
| Temporal specialist, two-image | `TERRASCOPE_URL` | `POST /run` | `services/terrascope/teres/` | `modal_app.py` |

The old `backend/services/mdetr/O2_RTDETR_endpoint/` is another deployable detector with its own Modal wrapper. The current agent's `object_detection` tool calls `ground_service`, so deploying O2-RTDETR alone will **not** redirect that tool. The duplicate top-level `services/O2_RTDETR_endpoint/` was removed.

`geo_analysis_agent` runs calculations in Python, and Qwen-Agent's built-in `image_zoom_in_tool`, `image_search`, and `web_search` are not backed by a service under `backend/services/`. They therefore have no Modal URL setting in the table.

The old frontend `/api/detect` route reads `YOLOVR_API_URL` in the **Next.js** process; the current chat does not call that route. Setting `YOLOVR_URL` in FastAPI controls the **agent tool**. You can point both variables at the same service URL. Similarly, the old direct frontend EarthMind and TerraScope routes have their own Next.js variables; they do not configure the agent.

## After deploying a service

Set its returned URL in the terminal that starts FastAPI. For example, in PowerShell:

```powershell
$env:QWEN_URL = "https://<your-qwen-service>/v1"
$env:MOONDREAM_URL = "https://<your-moondream-service>"
$env:GROUND_URL = "https://<your-perception-service>"
$env:YOLOVR_URL = "https://<your-yolo-service>"
$env:SAM3_URL = "https://<your-sam3-service>"
$env:OCR_URL = "https://<your-ocr-service>"
$env:GSD_URL = "https://<your-gsd-service>"
$env:ENHANCER_URL = "https://<your-enhancer-service>"
$env:CONTEXT_URL = "https://<your-location-service>"
$env:EARTHMIND_URL = "https://<your-earthmind-service>"
$env:EARTHDIAL_URL = "https://<your-earthdial-service>"
$env:TERRASCOPE_URL = "https://<your-terrascope-service>"
python -m uvicorn backend.app.app:app --host 127.0.0.1 --port 8080
```

Set only the services you have deployed; omitted variables keep the defaults in `common.py`. Replace every placeholder before running the command. A source directory or Modal script in the repo does not mean its endpoint is deployed or reachable. Test each deployed `/health` or `/docs` endpoint where provided, then try the corresponding chat task. Never put service-account private keys in a committed file.

The YOLO-World `/predict` API returns an annotated image and **does not return numeric detections**. The grounding agent can report its visual output, while `object_detection` and `open_language_grounding` remain the tools for coordinates.
