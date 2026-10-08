# SAC GeoAI: EarthDial & Ground Service Deployable Package

This package contains the complete, production-ready microservices configured for local execution and Modal serverless cloud deployment.

## Directory Structure
- Ground_service/perception_backend/:
  - pp/api/app.py: FastAPI application serving Object Detection (YOLOv8-OBB), Segmentation (SegEarth/FastSAM), and Image Analyzer.
  - deployment/modal/perception_tools_modal.py: Modal serverless deployment script (T4 GPU).
  - 
un_server.py: Local FastAPI launcher (http://localhost:8000).
  - weights/: Model weights (yolov8n-obb.pt, segearth_ov3.pt).
  - packages/sac-shared-contracts/: Standardized Pydantic schemas.

- Earth_dial/:
  - 	emporal_comparison_tool/:
    - pp.py & server.py: FastAPI application serving bi-temporal satellite comparison.
    - modal_app.py: Modal serverless deployment script (A10G GPU).
    - 
un_server.py: Local FastAPI launcher (http://localhost:8001).
  - EarthDial/src/: Core neural architecture and model implementation.

---

## Deployment to Modal Cloud

### 1. Deploy Ground Service (Perception)
`ash
cd Ground_service/perception_backend
modal deploy deployment/modal/perception_tools_modal.py
`
* Deploy target: https://<workspace>--sac-geoai-perception-api.modal.run
* Endpoints: /object_detector, /segmentation_grounding, /image_analyzer, /health, /docs

### 2. Deploy EarthDial (Temporal Comparison)
`ash
cd Earth_dial/temporal_comparison_tool
modal deploy modal_app.py
`
* Deploy target: https://<workspace>--earthdial-temporal-service-earthdialmodelservice-web-app.modal.run
* Endpoints: /compare, /compare/upload, /model/chat, /health, /docs

---

## Local FastAPI Execution

### Run Ground Service locally:
`ash
cd Ground_service/perception_backend
python run_server.py
`
* Accessible at: http://localhost:8000/docs

### Run EarthDial locally:
`ash
cd Earth_dial/temporal_comparison_tool
python run_server.py
`
* Accessible at: http://localhost:8001/docs
