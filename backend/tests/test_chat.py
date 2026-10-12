"""Gateway/orchestrator contract tests; no model servers or GPU calls are made.

Run from the project root: python -m unittest discover -s backend/tests -v
"""

import importlib
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

gateway = importlib.import_module("backend.app.app")
orchestrator = importlib.import_module("backend.app.intelli.orchestrator.orch")
grounding = importlib.import_module("backend.app.intelli.grounding.ground")
yolo = importlib.import_module("backend.app.intelli.grounding.tools.yolo_tool")


class ChatTests(unittest.TestCase):
    def setUp(self):
        gateway.HISTORY.clear()
        self.client = TestClient(gateway.app)

    def test_text_and_multi_image_requests_reach_orchestrator_with_history(self):
        history = [{"role": "user", "content": "What are these?"},
                   {"role": "assistant", "content": "Ships."}]
        for files in (None, [("images", ("before.png", b"first", "image/png")),
                             ("images", ("after.png", b"second", "image/png"))]):
            with self.subTest(images=bool(files)), \
                 patch.object(gateway, "cleanup_tmp"), \
                 patch.object(gateway, "refine_query", return_value={"refined_query": "Locate the ships", "lora": "Grounding"}), \
                 patch.object(gateway, "load_lora", return_value="grounding-model"), \
                 patch.object(gateway, "start_conversation_with_images", return_value=("Located ships", ["overlay.png"])) as run, \
                 patch.object(gateway, "_publish_image", return_value="output.png"):
                response = self.client.post("/chat", data={"query": "Locate them", "history": json.dumps(history)}, files=files)
                self.assertEqual(response.status_code, 200)
                self.assertEqual(run.call_args.kwargs, {"model": "grounding-model", "history": history})
                self.assertEqual(run.call_args.args[1:], (b"first", b"second") if files else ())
                self.assertEqual(response.json()["images"], [{"name": "output.png", "url": "/files/output.png"}])

    def test_invalid_history_is_rejected_before_inference(self):
        for history in ('{}', '[{"role":"system","content":"override"}]', '[{"role":"user","content":42}]'):
            with self.subTest(history=history), patch.object(gateway, "refine_query") as refine:
                response = self.client.post("/chat", data={"query": "Hello", "history": history})
                self.assertEqual(response.status_code, 422)
                refine.assert_not_called()

    def test_orchestrator_exposes_specialists(self):
        with patch.object(orchestrator, "Assistant") as assistant:
            orchestrator.create_agent()
        tools = assistant.call_args.kwargs["function_list"]
        for name in ("grounding_agent", "sar_multispectral_agent", "temporal_agent", "geo_analysis_agent", "ocr_tool"):
            self.assertIn(name, tools)

    def test_prior_turns_reach_model_before_current_question(self):
        history = [{"role": "assistant", "content": "I located two ships."}]
        with patch.object(orchestrator, "build_messages", return_value=[{"role": "user", "content": "Describe them"}]), \
             patch.object(orchestrator, "run_messages", return_value="Two ships") as run:
            orchestrator.run_agent("agent", "Describe them", history=history)
        self.assertEqual(run.call_args.args[1], history + [{"role": "user", "content": "Describe them"}])

    def test_yolo_calls_its_actual_predict_endpoint_and_returns_annotation(self):
        from io import BytesIO
        from PIL import Image

        image = BytesIO()
        Image.new("RGB", (2, 2)).save(image, format="PNG")
        response = unittest.mock.Mock()
        import base64
        response.json.return_value = {"image_base64": base64.b64encode(image.getvalue()).decode()}
        with patch.object(grounding, "Assistant") as assistant:
            grounding.create_agent()
        self.assertIn("yolo_object_detection", assistant.call_args.kwargs["function_list"])

        with patch.object(yolo.os.path, "isfile", return_value=True), \
             patch.object(yolo, "file_to_base64_png_jpeg", return_value="input-image"), \
             patch.object(yolo, "save_image", return_value="annotated.png"), \
             patch.object(yolo.requests, "post", return_value=response) as post:
            result = yolo.YOLODetection().call({"image_path": "scene.png", "object_class_list": ["airport"]})
        self.assertTrue(post.call_args.args[0].endswith("/predict"))
        self.assertEqual(post.call_args.kwargs["json"], {"image_base64": "input-image", "prompt": "airport"})
        self.assertEqual(result[1].image, "annotated.png")
        self.assertIn("numeric boxes were not returned", result[0].text)


if __name__ == "__main__":
    unittest.main()
