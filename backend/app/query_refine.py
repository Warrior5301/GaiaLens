"""Translate a user question and recent history into a task and optional LoRA choice."""

import requests

from backend.app.intelli.common import MOONDREAM_URL


def refine_query(query: str, image_base64: str | None = None, history: list[dict] | None = None) -> dict:
    """
    Call the Moondream /run endpoint, loading the model first if needed.

    `history` is a list of {"role": ..., "content": ...} turns.
    Returns {"related_to_history", "intent", "refined_query", "lora"}.
    """
    payload = {"query": query, "image_base64": image_base64, "history": history}
    response = requests.post(f"{MOONDREAM_URL}/run", json=payload, timeout=300)
    # This service uses 409 to indicate that its model must be loaded before retrying.
    if response.status_code == 409:
        requests.post(f"{MOONDREAM_URL}/load", timeout=900).raise_for_status()
        response = requests.post(f"{MOONDREAM_URL}/run", json=payload, timeout=300)
    response.raise_for_status()
    return response.json()
