"""Test the live Moondream API using the shared base64 image."""

import base64
import os
from pathlib import Path

import requests


# Set this to the exact *.modal.run URL printed by modal serve modal_app.py.
# Never use the Modal dashboard URL (/apps/...).
BASE_URL = " https://rajputvansh4391--gaialens-moondream-fastapi-app-dev.modal.run"
IMAGE_PATH = Path(__file__).resolve().parents[1] / "earthmind" / "base64_image.txt"
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 900


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
    try:
        response = requests.post(
            url,
            json=payload,
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Request to {url} timed out or could not connect: {exc}. "
            "Check that modal serve is still running and retry; the first cold start can take several minutes."
        ) from exc
    if not response.ok:
        raise RuntimeError(
            f"POST {url} failed with HTTP {response.status_code}:\n{response.text}"
        )
    return response.json()


def main() -> None:
    if not BASE_URL:
        raise SystemExit(
            "Set MOONDREAM_MODAL_URL to the *.modal.run URL printed by "
            "modal serve modal_app.py before running this test."
        )
    if "modal.com/apps/" in BASE_URL:
        raise SystemExit(
            "MOONDREAM_MODAL_URL is a Modal dashboard URL. Use the *.modal.run "
            "web endpoint printed by modal serve modal_app.py."
        )

    image_base64 = read_image_base64(IMAGE_PATH)

    try:
        health = requests.get(
            f"{BASE_URL}/health",
            timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
        )
    except requests.RequestException as exc:
        raise SystemExit(
            f"Could not get a health response from {BASE_URL}/health: {exc}\n"
            "The Modal container may still be starting. Keep modal serve running and try again."
        ) from exc
    if not health.ok:
        raise RuntimeError(
            f"GET {BASE_URL}/health failed with HTTP {health.status_code}:\n"
            f"{health.text}"
        )
    print("API health:", health.json())

    print("Loading Moondream into the Modal GPU container. First load may take a while...")
    print("Load result:", post_json(f"{BASE_URL}/load"))

    payload = {
        "query": "What do you see in this image?",
        "image_base64": image_base64,
        "history": [],
    }
    print("Inference result:")
    print(post_json(f"{BASE_URL}/run", payload))


if __name__ == "__main__":
    main()
