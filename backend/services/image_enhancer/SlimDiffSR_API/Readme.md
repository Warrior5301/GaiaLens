# SlimDiffSR FastAPI Service

FastAPI wrapper for the official [SlimDiffSR](https://github.com/wwangcece/SlimDiffSR) super-resolution model.

The purpose of this service is to expose SlimDiffSR as an HTTP API so that an AI agent/orchestrator can call super-resolution as a tool.

## Architecture

```text
AI Agent / Orchestrator
          |
          | HTTP POST + JSON
          v
  FastAPI SlimDiffSR API
          |
          v
 Official SlimDiffSR Inference
          |
          v
   Super-Resolved Image
          |
          | JSON + Base64
          v
 AI Agent / Downstream VLM
```

The API uses **Base64-encoded images inside JSON**.

There is currently **no multipart/file-upload endpoint**.

---

# 1. Requirements

### Hardware

A CUDA-enabled NVIDIA GPU is recommended.

The model was tested on:

- NVIDIA RTX 6000 Ada
- 48 GB VRAM

### Software

- Python 3.10
- CUDA-compatible PyTorch
- Linux recommended

---

# 2. Repository Setup

This API is a wrapper around the official SlimDiffSR repository.

Clone the official repository:

```bash
git clone https://github.com/wwangcece/SlimDiffSR.git
cd SlimDiffSR
```

Then copy the provided `app/` and `weight/` folders into the repository.

The final structure should look like:

```text
SlimDiffSR/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   └── sr_service.py
│
├── weight/
│   └── net_params_4x.pkl
│
├── requirements.txt
├── requirements-api.txt
│
├── Student/
├── bsr/
├── ram/
├── utils_/
├── model.py
├── forward.py
├── test_ours.py
└── ...
```

The folders/files such as `Student/`, `bsr/`, `ram/`, etc. come from the official SlimDiffSR repository.

---

# 3. Create Virtual Environment

From the SlimDiffSR directory:

```bash
python3.10 -m venv .venv
```

Activate it.

### Linux

```bash
source .venv/bin/activate
```

### Windows PowerShell

```powershell
.venv\Scripts\activate
```

You should see something similar to:

```text
(.venv)
```

at the beginning of your terminal prompt.

---

# 4. Install Dependencies

Install the SlimDiffSR/model dependencies:

```bash
pip install -r requirements.txt
```

Install the API dependencies:

```bash
pip install -r requirements-api.txt
```

The API dependency file contains:

```text
fastapi==0.142.2
uvicorn[standard]==0.54.0
```

The model dependencies are kept separately because FastAPI/Uvicorn are the API layer, while PyTorch, Transformers, Diffusers, etc. are required by SlimDiffSR itself.

---

# 5. Model Weights

The SlimDiffSR checkpoint must be present at:

```text
weight/net_params_4x.pkl
```

The checkpoint is loaded when the `/load` endpoint is called.

The checkpoint is **not** part of the FastAPI code itself.

---

# 6. SD-Turbo Dependency

The SlimDiffSR inference code uses the following components from:

```text
stabilityai/sd-turbo
```

Specifically:

```text
tokenizer/
text_encoder/
```

These are used to create the text embedding required by the SlimDiffSR model.

If Hugging Face authentication or network access is required in your environment, configure it before starting the service.

---

# 7. Configuration

The API supports configuration through environment variables.

Typical configuration:

```text
MODEL_PATH=weight/net_params_4x.pkl
SD_PATH=stabilityai/sd-turbo
TAESD_PATH=madebyollin/taesd
UPSCALE=4
PRUNE_DEPTH=2
FP16=false
```

For example, on Linux:

```bash
export SLIMDIFFSR_MODEL_PATH=weight/net_params_4x.pkl
export SLIMDIFFSR_UPSCALE=4
export SLIMDIFFSR_FP16=false
```

For the validated setup, use:

```bash
export SLIMDIFFSR_FP16=false
```

---

# 8. Start the API Server

From the root of the SlimDiffSR repository:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

### Why one worker?

The SlimDiffSR model is loaded into GPU memory.

Using multiple Uvicorn workers can create multiple processes, potentially causing multiple copies of the model to occupy GPU memory.

Therefore, use:

```bash
--workers 1
```

for this service.

---

# 9. Check Whether the Server Is Running

Once the server starts, test:

```bash
curl http://localhost:8000/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": false
}
```

After loading the model, it should report:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

# 10. Interactive API Documentation

FastAPI automatically generates interactive API documentation.

Open:

```text
http://localhost:8000/docs
```

This provides Swagger UI, where you can directly test the endpoints using **Try it out → Execute**.

FastAPI also provides ReDoc at:

```text
http://localhost:8000/redoc
```

These are automatically generated from the API's OpenAPI schema.

If the server is running on another machine, replace `localhost` with that machine's IP address:

```text
http://<SERVER_IP>:8000/docs
```

---

# 11. API Endpoints

The service currently provides:

| Method | Endpoint         | Purpose                      |
| ------ | ---------------- | ---------------------------- |
| GET    | `/health`        | Check server/model status    |
| GET    | `/ready`         | Check whether model is ready |
| POST   | `/load`          | Load SlimDiffSR into GPU     |
| POST   | `/unload`        | Unload SlimDiffSR from GPU   |
| POST   | `/super-resolve` | Super-resolve an image       |

---

# 12. `/health`

## Method

```text
GET /health
```

## Command

```bash
curl http://localhost:8000/health
```

## Example response

```json
{
  "status": "ok",
  "model_loaded": false
}
```

After loading:

```json
{
  "status": "ok",
  "model_loaded": true
}
```

This endpoint is useful for checking whether the API server is alive.

---

# 13. `/load`

## Method

```text
POST /load
```

Loads the SlimDiffSR model and required text encoder into GPU memory.

## Command

```bash
curl -X POST http://localhost:8000/load
```

Example response:

```json
{
  "status": "loaded"
}
```

You should call this before using `/super-resolve`.

The model remains loaded in GPU memory so subsequent requests do not need to reload the model.

---

# 14. `/ready`

## Method

```text
GET /ready
```

Checks whether the model is loaded and ready for inference.

## Command

```bash
curl http://localhost:8000/ready
```

If ready:

```json
{
  "status": "ready"
}
```

If the model has not been loaded, the endpoint returns HTTP `503`.

Typical workflow:

```text
Start server
     |
     v
GET /health
     |
     v
POST /load
     |
     v
GET /ready
     |
     v
POST /super-resolve
```

---

# 15. `/super-resolve`

This is the main endpoint used by the AI agent.

## Method

```text
POST /super-resolve
```

## Content-Type

```text
application/json
```

## Request body

```json
{
  "image_base64": "<BASE64_ENCODED_IMAGE>"
}
```

The image should be converted to Base64 before being sent.

---

# 16. Convert Image to Base64

### Linux

For example:

```bash
base64 -w 0 satellite.png > image.b64
```

### macOS

```bash
base64 satellite.png | tr -d '\n' > image.b64
```

### Windows PowerShell

```powershell
[Convert]::ToBase64String(
    [IO.File]::ReadAllBytes("satellite.png")
) | Out-File image.b64 -NoNewline
```

---

# 17. Create API Request Using Python

The easiest method is to use Python.

```python
import base64
import json

with open("satellite.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")

request = {
    "image_base64": image_base64
}

with open("request.json", "w") as f:
    json.dump(request, f)
```

This creates:

```text
request.json
```

containing:

```json
{
  "image_base64": "iVBORw0KGgoAAAANSUhEUg..."
}
```

---

# 18. Call `/super-resolve` Using cURL

Once `request.json` has been created:

```bash
curl -X POST http://localhost:8000/super-resolve \
     -H "Content-Type: application/json" \
     -d @request.json
```

The API returns JSON containing the super-resolved image as Base64.

---

# 19. Response Format

Example:

```json
{
  "image_base64": "<BASE64_SUPER_RESOLVED_IMAGE>",
  "filename": "super_resolved.png",
  "width": 4096,
  "height": 2748,
  "scale": 4,
  "input_width": 1024,
  "input_height": 687,
  "processing_time_seconds": 5.458
}
```

### Response fields

| Field                     | Meaning                                |
| ------------------------- | -------------------------------------- |
| `image_base64`            | Super-resolved image encoded as Base64 |
| `filename`                | Output filename                        |
| `width`                   | Output image width                     |
| `height`                  | Output image height                    |
| `scale`                   | Upscaling factor                       |
| `input_width`             | Original input width                   |
| `input_height`            | Original input height                  |
| `processing_time_seconds` | Inference time                         |

---

# 20. Save the Returned Image

The returned `image_base64` must be decoded back into image bytes.

```python
import base64

output_bytes = base64.b64decode(result["image_base64"])

with open("super_resolved.png", "wb") as f:
    f.write(output_bytes)
```

---

# 21. Complete Python Client

This is the recommended way for another application to use the API.

```python
import base64
import requests

API_URL = "http://localhost:8000"

# --------------------------------------------------
# 1. Load model
# --------------------------------------------------

response = requests.post(f"{API_URL}/load")
response.raise_for_status()

print("Model loaded")


# --------------------------------------------------
# 2. Read and encode image
# --------------------------------------------------

with open("satellite.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode("utf-8")


# --------------------------------------------------
# 3. Send image to SlimDiffSR
# --------------------------------------------------

response = requests.post(
    f"{API_URL}/super-resolve",
    json={
        "image_base64": image_base64
    }
)

response.raise_for_status()

result = response.json()


# --------------------------------------------------
# 4. Print information
# --------------------------------------------------

print("Input size:")
print(result["input_width"], "x", result["input_height"])

print("Output size:")
print(result["width"], "x", result["height"])

print("Scale:", result["scale"])

print(
    "Processing time:",
    result["processing_time_seconds"],
    "seconds"
)


# --------------------------------------------------
# 5. Decode output image
# --------------------------------------------------

output_bytes = base64.b64decode(
    result["image_base64"]
)

with open("super_resolved.png", "wb") as f:
    f.write(output_bytes)

print("Saved: super_resolved.png")
```

---

# 22. `/unload`

## Method

```text
POST /unload
```

Removes the model from GPU memory.

## Command

```bash
curl -X POST http://localhost:8000/unload
```

Example:

```json
{
  "status": "unloaded"
}
```

After unloading, `/ready` will no longer report the service as ready.

The model can be loaded again:

```bash
curl -X POST http://localhost:8000/load
```

---

# 23. Complete API Workflow

The normal workflow is:

### Terminal 1 — Start server

```bash
cd SlimDiffSR
source .venv/bin/activate

export SLIMDIFFSR_FP16=false

uvicorn app.main:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1
```

### Terminal 2 — Check server

```bash
curl http://localhost:8000/health
```

### Load model

```bash
curl -X POST http://localhost:8000/load
```

### Check readiness

```bash
curl http://localhost:8000/ready
```

### Send image

```bash
curl -X POST http://localhost:8000/super-resolve \
     -H "Content-Type: application/json" \
     -d @request.json
```

### Unload when finished

```bash
curl -X POST http://localhost:8000/unload
```

---

# 24. Using the API from Another Machine

If the FastAPI server is running on:

```text
192.168.1.100
```

port:

```text
8000
```

then another machine can call:

```bash
curl http://192.168.1.100:8000/health
```

and:

```bash
curl -X POST http://192.168.1.100:8000/load
```

The super-resolution request becomes:

```python
import requests

response = requests.post(
    "http://192.168.1.100:8000/super-resolve",
    json={
        "image_base64": image_base64
    }
)

result = response.json()
```

The server's port must be reachable from the client machine.

---

# 25. AI Agent Integration

The intended architecture is:

```text
                 AI Agent
                     |
                     | decides that
                     | super-resolution is required
                     v
             super_resolve tool
                     |
                     | HTTP POST
                     v
          SlimDiffSR FastAPI
                     |
                     v
          Official SlimDiffSR
                     |
                     v
              SR Image
                     |
                     | Base64
                     v
                 AI Agent
                     |
                     v
             Downstream VLM
```

The agent should treat SlimDiffSR as an external tool.

## Tool definition

A tool can be described as:

```python
{
    "name": "super_resolve",
    "description": (
        "Upscales a satellite image by 4x using SlimDiffSR. "
        "Use this when higher spatial resolution is required "
        "for downstream image analysis."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "image_base64": {
                "type": "string",
                "description": "Base64 encoded satellite image"
            }
        },
        "required": ["image_base64"]
    }
}
```

When the agent decides to call the tool, the tool executor sends:

```python
requests.post(
    "http://<SERVER_IP>:8000/super-resolve",
    json={
        "image_base64": image_base64
    }
)
```

The returned `image_base64` can then be passed to the next model/tool.

---

# 26. Important Design Decision

This API does **not** modify the official SlimDiffSR inference pipeline.

The API wrapper performs:

```text
Input Image
     ↓
Tensor conversion
     ↓
Official SlimDiffSR preprocessing
     ↓
Official SlimDiffSR model
     ↓
Official post-processing
     ↓
Output Image
     ↓
Base64
```

No custom image tiling is introduced by the API.

---

# 27. Validation

The API was validated against the official `test_ours.py` inference.

Test input:

```text
1024 × 687
```

Expected 4× output:

```text
4096 × 2748
```

Comparison between official inference and API output:

```text
MAE              = 0.0
RMSE             = 0.0
Maximum difference = 0
Exact pixel match  = 100%
```

Therefore, for the tested image, the API reproduces the official inference output exactly.

---

# 28. Troubleshooting

## API does not start

Check that the virtual environment is active:

```bash
which python
```

and:

```bash
which uvicorn
```

Then check:

```bash
pip show fastapi
pip show uvicorn
```

---

## Model is not loading

Check the checkpoint:

```bash
ls -lh weight/net_params_4x.pkl
```

Expected file:

```text
weight/net_params_4x.pkl
```

Also check that CUDA is available:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU')"
```

---

## `/super-resolve` returns "Model is not loaded"

Call:

```bash
curl -X POST http://localhost:8000/load
```

Then:

```bash
curl http://localhost:8000/ready
```

Only call `/super-resolve` after the service reports that it is ready.

---

## GPU out-of-memory

First check GPU usage:

```bash
nvidia-smi
```

Make sure only the intended SlimDiffSR process is occupying GPU memory.

Also make sure the server is running with:

```bash
--workers 1
```

---

# 29. API Summary

```text
GET  /health
     ↓
Check server status

POST /load
     ↓
Load model into GPU

GET  /ready
     ↓
Check model readiness

POST /super-resolve
     ↓
Base64 image → SlimDiffSR → Base64 SR image

POST /unload
     ↓
Release GPU memory
```

---

# 30. Quickest Way to Test Everything

Start the server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1
```

Open:

```text
http://localhost:8000/docs
```

Then:

```text
1. POST /load
       ↓
2. GET /ready
       ↓
3. POST /super-resolve
       ↓
4. Receive Base64 output
       ↓
5. POST /unload
```

The `/docs` page can be used to interactively test the API without writing a client.
