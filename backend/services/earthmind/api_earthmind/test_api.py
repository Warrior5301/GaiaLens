"""Test the live EarthMind Modal API using the shared base64 image."""

import base64
import os
from pathlib import Path

import requests


# Set this to the exact *.modal.run URL printed by modal serve modal_app.py.
# Do not use the Modal dashboard URL (/apps/...).
MODAL_URL = os.getenv("EARTHMIND_MODAL_URL", "").strip().rstrip("/")
IMAGE_PATH = Path(__file__).resolve().parents[1] / "base64_image.txt"
TIMEOUT_SECONDS = 900


def read_image_base64(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Base64 image file not found: {path}")
    encoded = path.read_text(encoding="utf-8").strip()
    if encoded.startswith("data:"):
        header, separator, encoded = encoded.partition(",")
        if not separator or ";base64" not in header:
            raise ValueError("Image data URL must use base64 encoding.")
    base64.b64decode(encoded, validate=True)
    return encoded


def post_json(url: str, payload: dict | None = None) -> dict:
    response = requests.post(url, json=payload, timeout=TIMEOUT_SECONDS)
    if not response.ok:
        raise RuntimeError(
            f"POST {url} failed with HTTP {response.status_code}:\n{response.text}"
        )
    return response.json()


def main() -> None:
    if not MODAL_URL:
        raise SystemExit(
            "Set EARTHMIND_MODAL_URL to the *.modal.run URL printed by "
            "modal serve modal_app.py before running this test."
        )
    if "modal.com/apps/" in MODAL_URL:
        raise SystemExit(
            "EARTHMIND_MODAL_URL is a Modal dashboard URL. Use the *.modal.run "
            "web endpoint printed by modal serve modal_app.py."
        )

    image_base64 = read_image_base64(IMAGE_PATH)

    health = requests.get(f"{MODAL_URL}/health", timeout=30)
    if not health.ok:
        raise RuntimeError(
            f"GET {MODAL_URL}/health failed with HTTP {health.status_code}:\n"
            f"{health.text}"
        )
    print("API health:", health.json())

    print("Loading model into the Modal GPU container (first call may take a while)...")
    print("Load result:", post_json(f"{MODAL_URL}/load"))

    payload = {
        "query_type": "normal",
        "prompt": "What do you see in this image?",
        "max_new_tokens": 128,
        "image_base64": image_base64,
    }
    result = post_json(f"{MODAL_URL}/run", payload)
    print("Inference result:")
    print(result)


if __name__ == "__main__":
    main()

