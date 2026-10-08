"""
Comprehensive Automated Test Suite for EarthDial Model & Temporal Comparison Tool
==================================================================================
"""

import base64
import io
import os
import sys
import unittest
from pathlib import Path
from typing import Tuple

# Ensure fast deterministic test execution without waiting on 3GB weights download
os.environ["EARTHDIAL_MOCK_MODE"] = "true"

from PIL import Image, ImageDraw

# Add current folder to sys.path
tool_dir = Path(__file__).resolve().parent
if str(tool_dir) not in sys.path:
    sys.path.insert(0, str(tool_dir))

from model import EarthDialModel
from tool import TemporalComparisonTool
from schemas import PromptStyle, TemporalComparisonRequest
from app import app
from starlette.testclient import TestClient


def create_synthetic_temporal_pair() -> Tuple[Path, Path]:
    """Generates a synthetic before-and-after pair of satellite images for testing."""
    test_data_dir = tool_dir / "test_data"
    test_data_dir.mkdir(parents=True, exist_ok=True)

    img1_path = test_data_dir / "satellite_t1_baseline.png"
    img2_path = test_data_dir / "satellite_t2_modified.png"

    # Image 1 (T1 Baseline): Green terrain with a river
    im1 = Image.new("RGB", (512, 512), color=(45, 120, 50))
    d1 = ImageDraw.Draw(im1)
    d1.line([(0, 200), (200, 250), (350, 220), (512, 300)], fill=(30, 90, 180), width=35)
    im1.save(img1_path)

    # Image 2 (T2 Modified): Same terrain with new road and building structures
    im2 = im1.copy()
    d2 = ImageDraw.Draw(im2)
    d2.rectangle([(100, 80), (220, 180)], fill=(180, 180, 185), outline=(50, 50, 50), width=3)
    d2.line([(160, 180), (180, 380), (512, 420)], fill=(120, 100, 80), width=20)
    im2.save(img2_path)

    return img1_path, img2_path


class TestEarthDialSuite(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.img1_path, cls.img2_path = create_synthetic_temporal_pair()
        cls.model = EarthDialModel(mock_mode=True)
        cls.tool = TemporalComparisonTool(model=cls.model)

    def test_01_model_direct_inference(self):
        """Verify EarthDialModel direct inference and get_info."""
        info = self.model.get_info()
        self.assertIn("model_path", info)
        self.assertIn("device", info)
        self.assertEqual(info["device"], "mock")

        # Test single image direct generation
        img1 = Image.open(self.img1_path)
        text, meta, dev = self.model.generate(
            images=[img1],
            prompt="<image>\nWhat is visible in this satellite patch?"
        )
        self.assertIsInstance(text, str)
        self.assertGreater(len(text), 5)
        self.assertEqual(len(meta), 1)

    def test_02_image_loading_and_conversion(self):
        """Verify image loader handles PIL, Path, bytes, and base64 strings."""
        # 1. From Path
        img_from_path = self.tool.load_image(self.img1_path)
        self.assertIsInstance(img_from_path, Image.Image)
        self.assertEqual(img_from_path.size, (512, 512))

        # 2. From PIL Image directly
        img_pil = self.tool.load_image(img_from_path)
        self.assertEqual(img_pil.size, img_from_path.size)
        self.assertEqual(img_pil.mode, "RGB")

        # 3. From base64 string
        buf = io.BytesIO()
        img_from_path.save(buf, format="PNG")
        b64_str = f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode('utf-8')}"
        img_from_b64 = self.tool.load_image(b64_str)
        self.assertIsInstance(img_from_b64, Image.Image)
        self.assertEqual(img_from_b64.size, (512, 512))

    def test_03_prompt_formatting(self):
        """Verify prompt formatting for different image counts and styles."""
        # Evaluation format for 2 images
        prompt_2 = self.tool.format_prompt(
            num_images=2,
            query="Are changes detected?",
            style=PromptStyle.EVALUATION
        )
        self.assertIn("[changedet] [hr_rgb_temp_0.5]", prompt_2)
        self.assertEqual(prompt_2.count("<image>"), 2)

        # Conversational format for 3 images
        prompt_conv = self.tool.format_prompt(
            num_images=3,
            query="Compare Image-1 with Image-3.",
            style=PromptStyle.CONVERSATIONAL
        )
        self.assertIn("Image-1: <image>\nImage-2: <image>\nImage-3: <image>\n", prompt_conv)

    def test_04_tool_comparison_execution(self):
        """Verify direct Python tool comparison execution."""
        result = self.tool.compare(
            images=[self.img1_path, self.img2_path],
            query="Identify new urban developments and road construction.",
            prompt_style=PromptStyle.EVALUATION
        )
        self.assertTrue(result.success)
        self.assertEqual(result.num_images, 2)
        self.assertGreater(len(result.response), 10)
        self.assertIn("urban", result.response.lower())
        self.assertGreater(result.execution_time_seconds, 0)
        self.assertEqual(len(result.image_metadata), 2)

    def test_05_validation_error_on_single_image(self):
        """Verify tool enforces at least 2 images for temporal comparison."""
        with self.assertRaises(ValueError):
            self.tool.compare(
                images=[self.img1_path],
                query="Compare"
            )

    def test_06_api_health_endpoint(self):
        """Test GET /health returns 200 with runtime diagnostics."""
        with TestClient(app) as client:
            resp = client.get("/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertEqual(data["status"], "healthy")
            self.assertIn("device", data)
            self.assertIn("cuda_available", data)

    def test_07_api_templates_endpoint(self):
        """Test GET /templates returns recommended domain templates."""
        with TestClient(app) as client:
            resp = client.get("/templates")
            self.assertEqual(resp.status_code, 200)
            templates = resp.json()["templates"]
            self.assertGreaterEqual(len(templates), 4)

    def test_08_api_compare_json_endpoint(self):
        """Test POST /compare with JSON payload (paths or base64)."""
        with TestClient(app) as client:
            payload = {
                "images": [str(self.img1_path), str(self.img2_path)],
                "query": "Assess structural damage and vegetation alterations.",
                "prompt_style": "evaluation",
                "temperature": 0.0,
                "max_new_tokens": 128
            }
            resp = client.post("/compare", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["num_images"], 2)
            self.assertIn("response", data)

    def test_09_api_compare_upload_multipart_endpoint(self):
        """Test POST /compare/upload with multipart form-data image uploads."""
        with TestClient(app) as client:
            with open(self.img1_path, "rb") as f1, open(self.img2_path, "rb") as f2:
                files = [
                    ("files", ("baseline.png", f1.read(), "image/png")),
                    ("files", ("modified.png", f2.read(), "image/png"))
                ]
                data = {
                    "query": "Identify new structures and roads.",
                    "prompt_style": "evaluation",
                    "temperature": "0.0"
                }
                resp = client.post("/compare/upload", files=files, data=data)
                self.assertEqual(resp.status_code, 200)
                json_data = resp.json()
                self.assertTrue(json_data["success"])
                self.assertEqual(json_data["num_images"], 2)
                self.assertIn("response", json_data)

    def test_10_api_model_chat_endpoint(self):
        """Test POST /model/chat for general model VQA inference."""
        with TestClient(app) as client:
            payload = {
                "images": [str(self.img1_path)],
                "prompt": "Describe this satellite observation.",
                "temperature": 0.0,
                "max_new_tokens": 128
            }
            resp = client.post("/model/chat", json=payload)
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["success"])
            self.assertEqual(data["num_images"], 1)
            self.assertIn("response", data)

    def test_11_api_model_info_endpoint(self):
        """Test GET /model/info returns model architecture and memory status."""
        with TestClient(app) as client:
            resp = client.get("/model/info")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertIn("architecture", data)
            self.assertIn("parameters", data)
            self.assertIn("model_path", data)


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestEarthDialSuite)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    sys.exit(0 if result.wasSuccessful() else 1)
