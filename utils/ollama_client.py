"""
utils/ollama_client.py

Thin wrapper around Ollama for all LLM calls (text + vision).
Pull models before use:
  ollama pull llama3.1:8b    (text)
  ollama pull llava           (vision)
"""

import ollama, os, base64
from dotenv import load_dotenv
load_dotenv()

MODEL        = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
VISION_MODEL = os.getenv("VISION_MODEL", "llava")


def chat(system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
    response = ollama.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_message},
        ],
        options={"temperature": temperature},
    )
    return response["message"]["content"].strip()


def describe_image(image_path: str, prompt: str = None) -> str:
    """
    Send an image to the Ollama vision model and get a text description.
    Uses the VISION_MODEL (e.g., llava, llama3.2-vision).
    """
    if prompt is None:
        prompt = (
            "You are analyzing a figure from a research paper. "
            "Describe this image in detail: what type of figure is it "
            "(chart, diagram, photo, table, etc.), what does it show, "
            "what are the key data points, labels, axes, or trends? "
            "Be specific and factual."
        )

    # Read image and encode to base64
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    response = ollama.chat(
        model=VISION_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
                "images": [image_data],
            }
        ],
        options={"temperature": 0.1},
    )
    return response["message"]["content"].strip()


def check_model():
    try:
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        ready = any(m.startswith(MODEL) for m in available)
        status = "ready" if ready else f"NOT found — run: ollama pull {MODEL}"
        print(f"[Ollama] Text model '{MODEL}': {status}")
        return ready
    except Exception as e:
        print(f"[Ollama] Connection failed: {e}")
        return False


def check_vision_model():
    try:
        models = ollama.list()
        available = [m["name"] for m in models.get("models", [])]
        ready = any(m.startswith(VISION_MODEL) for m in available)
        status = "ready" if ready else f"NOT found — run: ollama pull {VISION_MODEL}"
        print(f"[Ollama] Vision model '{VISION_MODEL}': {status}")
        return ready
    except Exception as e:
        print(f"[Ollama] Connection failed: {e}")
        return False
