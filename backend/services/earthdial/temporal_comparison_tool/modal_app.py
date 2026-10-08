"""
Modal Serverless Deployment for EarthDial Temporal Comparison
============================================================

Deploy EarthDial Temporal Comparison API on serverless A10G/A100 GPUs in the cloud.

Commands:
    # 1. Run inference directly on cloud GPU to test:
    modal run temporal_comparison_tool/modal_app.py

    # 2. Serve the interactive FastAPI documentation locally via tunnel:
    modal serve temporal_comparison_tool/modal_app.py

    # 3. Deploy the live web endpoint to production:
    modal deploy temporal_comparison_tool/modal_app.py
"""

import base64
import os
import sys
from pathlib import Path
import modal

current_dir = Path(__file__).resolve().parent
earthdial_src = current_dir.parent / "EarthDial" / "src"

app_name = "earthdial-temporal-service"
app = modal.App(app_name)

# Mount huggingface cache volume for fast persistent model loading
hf_volume = modal.Volume.from_name("hf-cache", create_if_missing=True)

earthdial_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install(
        "numpy<2",
        "torch==2.1.2",
        "torchvision==0.16.2",
        extra_index_url="https://download.pytorch.org/whl/cu121"
    )
    .pip_install(
        "transformers==4.37.2",
        "timm==0.9.12",
        "einops>=0.6.1",
        "einops-exts>=0.0.4",
        "opencv-python-headless==4.9.0.80",
        "pillow>=10.0.0",
        "fastapi>=0.110.0",
        "uvicorn>=0.28.0",
        "python-multipart>=0.0.9",
        "pydantic>=2.5.0",
        "huggingface_hub>=0.20.0",
        "requests>=2.31.0",
        "accelerate>=0.28.0",
        "sentencepiece>=0.1.99",
        "imageio>=2.30.0",
        "hf_xet",
    )
    .pip_install("numpy<2")
    .add_local_dir(str(earthdial_src), remote_path="/root/EarthDial/src")
    .add_local_dir(str(current_dir), remote_path="/root/temporal_comparison_tool")
)


@app.cls(
    image=earthdial_image,
    gpu="A10G",
    volumes={"/root/.cache/huggingface": hf_volume},
    timeout=1200,
    scaledown_window=300
)
class EarthDialModelService:
    @modal.enter()
    def startup(self):
        import sys
        if "/root/EarthDial/src" not in sys.path:
            sys.path.insert(0, "/root/EarthDial/src")
        if "/root/temporal_comparison_tool" not in sys.path:
            sys.path.insert(0, "/root/temporal_comparison_tool")

        from huggingface_hub import snapshot_download
        print("Ensuring EarthDial_4B_RGB weights are downloaded...")
        snapshot_download(
            repo_id="akshaydudhane/EarthDial_4B_RGB",
            ignore_patterns=["*.msgpack", "*.h5", "*.ot"]
        )
        hf_volume.commit()

        from comparator import EarthDialTemporalComparator
        import server

        print("Loading EarthDial model onto A10G GPU...")
        self.comparator = EarthDialTemporalComparator(
            checkpoint="akshaydudhane/EarthDial_4B_RGB",
            device="cuda",
            mock_mode=False,
            auto_device_map=True,
            lazy_load=False
        )
        server.comparator = self.comparator
        print("EarthDial model successfully loaded on GPU!")

    @modal.asgi_app()
    def web_app(self):
        import sys
        if "/root/EarthDial/src" not in sys.path:
            sys.path.insert(0, "/root/EarthDial/src")
        if "/root/temporal_comparison_tool" not in sys.path:
            sys.path.insert(0, "/root/temporal_comparison_tool")

        from server import app as fastapi_app
        return fastapi_app

    @modal.method()
    def run_comparison(
        self,
        images: list,
        query: str = "Analyze the images and describe any semantic or structural changes detected between the observations.",
        temperature: float = 0.0,
        max_new_tokens: int = 256
    ):
        """Direct programmatic invocation method for Python clients."""
        return self.comparator.compare(
            images=images,
            query=query,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )


@app.local_entrypoint()
def main():
    """Runs a live test of real EarthDial inference on Modal A10G GPU."""
    print("=" * 65)
    print(" Testing Real EarthDial Temporal Comparison Inference on Modal GPU")
    print("=" * 65)

    test_data_dir = current_dir / "test_data"
    img1_path = test_data_dir / "satellite_t1_baseline.png"
    img2_path = test_data_dir / "satellite_t2_modified.png"

    # If test images don't exist, create them
    if not img1_path.exists() or not img2_path.exists():
        from test_tool import create_synthetic_temporal_pair
        img1_path, img2_path = create_synthetic_temporal_pair()

    print(f"T1 Observation: {img1_path}")
    print(f"T2 Observation: {img2_path}")

    with open(img1_path, "rb") as f1, open(img2_path, "rb") as f2:
        b64_img1 = "data:image/png;base64," + base64.b64encode(f1.read()).decode()
        b64_img2 = "data:image/png;base64," + base64.b64encode(f2.read()).decode()

    service = EarthDialModelService()
    print("\nSending request to Modal GPU worker...")
    result = service.run_comparison.remote(
        images=[b64_img1, b64_img2],
        query="Analyze the images and identify all semantic and structural changes between the observations."
    )

    print("\n" + "=" * 65)
    print(" EarthDial Output on Cloud GPU:")
    print("=" * 65)
    print(f"Model Prompt : {result.formatted_prompt}")
    print(f"\nResponse     :\n{result.response}")
    print("-" * 65)
    print(f"Device Used  : {result.device}")
    print(f"Latency      : {result.execution_time_seconds}s")
    print(f"Tiles/Image  : {[m.num_tiles for m in result.image_metadata]}")
    print("=" * 65)
