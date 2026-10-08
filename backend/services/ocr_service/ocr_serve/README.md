# PP-OCRv5 Mobile FastAPI

A small API that accepts a base64-encoded image and returns recognized text using PaddleOCR's PP-OCRv5 Mobile detection and recognition models.

## Run

Use Python 3.10 or newer. PaddlePaddle installation can depend on your OS, Python version, and CPU/GPU setup; install a compatible PaddlePaddle 3.x build first using the [official installation guide](https://www.paddlepaddle.org.cn/install/quick).

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```

On first startup, PaddleOCR downloads the model files if they are not already cached. The API starts even if model initialization fails, so `/health` and `/ocr` return `503` until the model is ready; check the server logs for the initialization error.

## Endpoints

- `GET /health` — readiness check.
- `POST /ocr` — OCR an image; accepts raw base64 or a `data:image/...;base64,` data URL.
- `GET /docs` — interactive Swagger UI.

Example:

```powershell
$image = [Convert]::ToBase64String([IO.File]::ReadAllBytes("sample.png"))
$body = @{ image = $image } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/ocr -Method Post -ContentType "application/json" -Body $body
```

Response:

```json
{
  "text": "First recognized line\nSecond recognized line",
  "lines": ["First recognized line", "Second recognized line"]
}
```

Set `MAX_IMAGE_BYTES` to change the decoded image size limit (default: 10 MiB).
