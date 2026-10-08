# EarthDial Vision-Language & Temporal Comparison Service 🛰️⏱️

A clean, modular, and framework-agnostic toolkit and REST API for:
1. **Temporal Comparison Tool**: Multi-temporal remote-sensing image comparison and change detection ($\ge 2$ images).
2. **EarthDial Model**: Direct vision-language inference and conversational VQA ($0, 1,$ or multiple images).

Powered by **[EarthDial](https://github.com/hiyamdebary/EarthDial)** (`akshaydudhane/EarthDial_4B_RGB`, CVPR 2025).

---

## 📁 Modular Codebase Architecture

The codebase is organized into cleanly separated layers for maximum modularity and reusability:

```text
temporal_comparison_tool/
├── model.py            # EarthDialModel: Pure PyTorch/Transformers wrapper (load, generate, info)
├── tool.py             # TemporalComparisonTool: Multi-image comparison tool (load_image, format_prompt, compare)
├── schemas.py          # Pydantic v2 schemas for both Tool and Model endpoints
├── app.py              # Standard FastAPI service (Framework-agnostic REST API)
├── run.py              # Simple CLI server runner (python run.py --port 8000)
├── example_client.py   # Example script demonstrating requests calls to all endpoints
├── test_tool.py        # Automated unit test suite (11 passing tests)
├── requirements.txt    # Standard Python dependencies
├── test_data/          # Sample satellite image pairs (T1 and T2)
└── README.md           # Documentation & usage guide
```

---

## 🚀 Quickstart

### 1. Installation

```bash
cd temporal_comparison_tool
pip install -r requirements.txt
```

### 2. Start the REST API

Run locally on port 8000:
```bash
python run.py --port 8000
```

> **Tip for rapid offline testing:** Add `--mock` to start immediately in simulation mode without downloading model weights:
> ```bash
> python run.py --port 8000 --mock
> ```

Open interactive Swagger documentation in your browser:
👉 **`http://localhost:8000/docs`**

---

## 📡 REST API Endpoints Overview

| Category | Method | Path | Description |
| :--- | :--- | :--- | :--- |
| **Tool** | `POST` | `/compare` | Temporal comparison of 2+ images via JSON (file paths, URLs, base64) |
| **Tool** | `POST` | `/compare/upload` | Temporal comparison via direct multipart file upload |
| **Tool** | `GET` | `/templates` | Pre-configured prompt templates for change detection |
| **Model** | `POST` | `/model/chat` | Direct VQA or prompt inference with EarthDial on 0, 1, or more images |
| **Model** | `GET` | `/model/info` | Inspect model architecture, device, parameters, and VRAM |
| **System**| `GET` | `/health` | Hardware diagnostics and service health |

---

## 🛠️ API Usage Examples

### 1. Temporal Comparison Tool (`POST /compare`)

Compare two temporal satellite acquisitions:

```bash
curl -X POST "http://localhost:8000/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "images": [
      "test_data/satellite-image-t1.png",
      "test_data/satellite-image-t2.png"
    ],
    "query": "Are there any semantic changes detected in the provided images?",
    "prompt_style": "evaluation",
    "temperature": 0.0,
    "max_new_tokens": 128
  }'
```

**Response:**
```json
{
  "success": true,
  "query": "Are there any semantic changes detected in the provided images?",
  "formatted_prompt": "[changedet] [hr_rgb_temp_0.5] <image> <image>\n Are there any semantic changes detected in the provided images?",
  "response": "The area was urbanized.",
  "num_images": 2,
  "image_metadata": [
    {"index": 1, "width": 512, "height": 512, "num_tiles": 2},
    {"index": 2, "width": 512, "height": 512, "num_tiles": 2}
  ],
  "execution_time_seconds": 2.96,
  "device": "cuda",
  "model_name_or_path": "akshaydudhane/EarthDial_4B_RGB"
}
```

---

### 2. Direct Model Inference (`POST /model/chat`)

Query the model directly with a single image or multi-turn prompt:

```bash
curl -X POST "http://localhost:8000/model/chat" \
  -H "Content-Type: application/json" \
  -d '{
    "images": ["test_data/satellite-image-t1.png"],
    "prompt": "<image>\nDescribe the geographic terrain and land features.",
    "temperature": 0.0,
    "max_new_tokens": 128
  }'
```

---

### 3. Multipart File Upload (`POST /compare/upload`)

Upload image files directly from form data or frontend dashboards:

```bash
curl -X POST "http://localhost:8000/compare/upload" \
  -F "files=@test_data/satellite-image-t1.png" \
  -F "files=@test_data/satellite-image-t2.png" \
  -F "query=Identify all newly built infrastructure and roads."
```

---

## 🐍 Python Library Usage

You can also import and use the tool and model directly in your Python code:

### Using the Temporal Comparison Tool:
```python
from tool import TemporalComparisonTool, PromptStyle

# Initialize tool (loads model on demand)
tool = TemporalComparisonTool(checkpoint="akshaydudhane/EarthDial_4B_RGB")

# Compare two or more images
result = tool.compare(
    images=["image_2021.png", "image_2024.png"],
    query="Analyze the images and identify all semantic changes.",
    prompt_style=PromptStyle.EVALUATION
)

print("Change detection output:", result.response)
print("Latency:", result.execution_time_seconds, "seconds")
```

### Using the EarthDial Model directly:
```python
from PIL import Image
from model import EarthDialModel

# Initialize model wrapper
model = EarthDialModel(checkpoint="akshaydudhane/EarthDial_4B_RGB")

# Generate response for an image
image = Image.open("satellite_observation.png")
text, metadata, device = model.generate(
    images=[image],
    prompt="<image>\nWhat types of vegetation are visible?"
)

print("Model response:", text)
```

---

## 🧪 Testing

Run the automated test suite:
```bash
python test_tool.py
```

All 11 tests execute in under 1 second covering model inference, image loaders, prompt formatters, tool comparison, and all FastAPI endpoints.
