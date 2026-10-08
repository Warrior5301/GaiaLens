# EarthMind-4B FastAPI

This is a regular FastAPI ASGI application. It does not depend on Modal and can run under Uvicorn or another ASGI server. It preserves `GET /health`, `POST /v1/infer`, the four query types, base64 image inputs, and base64 PNG segmentation-mask outputs.

## Requirements

- Python 3.10
- A CUDA-capable NVIDIA GPU with BF16 support
- Enough host RAM and GPU memory for EarthMind-4B

The first inference downloads the pinned model revision into the Hugging Face cache at `~/.cache/earthmind` by default. Set `EARTHMIND_CACHE_DIR` to choose another cache path. You can override the model ID and revision with `EARTHMIND_MODEL_ID` and `EARTHMIND_MODEL_REVISION`.

## Run locally on Windows

From the project root in PowerShell:

```powershell
py -3.10 -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
python -m pip install -r api_earthmind/requirements.txt
python -m uvicorn app:app --app-dir api_earthmind --host 127.0.0.1 --port 8000
```

Open [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs), expand `POST /v1/infer`, and paste a request. For a deployed ASGI host, configure its process to run `uvicorn app:app` from the `api_earthmind` directory and provide the same model dependencies and a CUDA/BF16 GPU.

## Request types

Image fields accept raw base64 or a `data:image/...;base64,...` data URL. Each image is limited to 12 MiB. Temporal frames must be supplied in chronological order. Masks are binary arrays shaped `[region][row][column]`.

| `query_type` | Required input fields | Model path |
| --- | --- | --- |
| `normal` | Optional `image_base64`; without an image, the request is text-only | `predict_forward` |
| `temporal_analysis` | `images_base64` with 2 to 16 frames | `predict_forward(video=frames, ...)` |
| `multispectral_analysis` | `image_base64` and `rgb_image_base64` | `predict_forward_multi(image=..., rgb_image=..., ...)` |
| `masked_prompt` | `image_base64` and `mask_prompts` | `predict_forward(image=..., mask_prompts=...)` |

Example:

```json
{
  "query_type": "normal",
  "image_base64": "<base64-encoded image bytes>",
  "prompt": "Describe the land cover and visible features.",
  "max_new_tokens": 128
}
```

The response contains `prediction` and `prediction_masks`. Each mask is a base64-encoded PNG with width and height metadata. Masks are grouped as `[segmentation][frame]`; an empty list means no masks were returned.
