"""
ingestion/image_processor.py

Processes images extracted from research paper PDFs.
Uses Ollama's vision model (e.g., llava) to generate text
descriptions of figures, charts, and tables — these descriptions
become searchable chunks in the vector store.

Flow:
  Image → Vision LLM → text description → embedded as chunk
"""

import os
from typing import List, Dict
from utils.ollama_client import describe_image
from dotenv import load_dotenv

load_dotenv()

MIN_IMAGE_SIZE = int(os.getenv("MIN_IMAGE_SIZE", 100))  # Minimum px to filter decorative images


def _is_meaningful_image(width: int, height: int) -> bool:
    """Filter out tiny decorative images, icons, and logos."""
    if width < MIN_IMAGE_SIZE or height < MIN_IMAGE_SIZE:
        return False
    # Skip very narrow images (likely borders/lines)
    if width / max(height, 1) > 10 or height / max(width, 1) > 10:
        return False
    return True


def process_single_image(image_path: str, page: int, image_index: int, filename: str) -> Dict:
    """
    Send a single image to the vision model and return a description chunk.

    Returns:
        dict with keys: text, section, pages, type, image_path
    """
    print(f"  [Vision] Describing image {image_index} from page {page}...")

    try:
        description = describe_image(image_path)
    except Exception as e:
        print(f"  [Vision] Failed to describe image {image_index}: {e}")
        description = f"[Figure from page {page} — description unavailable]"

    # Create a chunk that looks like a text chunk but carries image metadata
    chunk = {
        "text": f"[Figure from page {page}] {description}",
        "section": f"Figure (page {page})",
        "pages": [page],
        "type": "image",
        "image_path": image_path,
    }
    return chunk


def process_images(images: List[Dict], session_id: str, filename: str) -> List[Dict]:
    """
    Batch-process all extracted images through the vision model.

    Args:
        images: list of {path, page, width, height, image_index}
        session_id: current session ID
        filename: source PDF filename

    Returns:
        List of description chunks ready for vector store ingestion
    """
    # Filter out small/decorative images
    meaningful = [
        img for img in images
        if _is_meaningful_image(img["width"], img["height"])
    ]

    if not meaningful:
        print(f"[ImageProcessor] No meaningful images found (all below {MIN_IMAGE_SIZE}px)")
        return []

    print(f"[ImageProcessor] Processing {len(meaningful)}/{len(images)} images "
          f"(filtered {len(images) - len(meaningful)} small/decorative)")

    description_chunks = []
    for img in meaningful:
        chunk = process_single_image(
            image_path=img["path"],
            page=img["page"],
            image_index=img["image_index"],
            filename=filename,
        )
        description_chunks.append(chunk)

    print(f"[ImageProcessor] Generated {len(description_chunks)} image descriptions")
    return description_chunks
