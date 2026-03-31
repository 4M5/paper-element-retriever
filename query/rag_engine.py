"""
query/rag_engine.py

Grounded RAG query engine for research papers.
Answers ONLY from retrieved chunks — cites section + page number.
Supports both text and image-description chunks (multimodal RAG).
"""

from ingestion.vector_store import retrieve
from utils.ollama_client import chat
from typing import Dict, List

SYSTEM_PROMPT = """You are a precise research assistant.
Answer the user's question using ONLY the paper excerpts provided below.

Rules:
1. Use ONLY information from the provided excerpts. Never use outside knowledge.
2. Always cite the source: mention the section name and page number.
   Example: "According to the Methods section (page 4)..."
3. If a figure or image description is provided, reference it naturally.
   Example: "As shown in the figure on page 3..."
4. If the answer is not in the excerpts, say exactly:
   "This information is not found in the uploaded paper."
5. Be precise and concise. Avoid padding or filler.
6. If multiple papers are uploaded, mention which paper each point comes from.
7. Never fabricate statistics, numbers, or author names.
"""


def _format_context(chunks: List[Dict]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        chunk_type = c.get("type", "text")

        if chunk_type == "image":
            lines.append(
                f"[Image/Figure {i} | Paper: {c['filename']} | "
                f"Page: {c['pages']} | Type: figure/image]"
            )
        else:
            lines.append(
                f"[Excerpt {i} | Paper: {c['filename']} | "
                f"Section: {c['section']} | Pages: {c['pages']}]"
            )

        lines.append(c["text"])
        lines.append("")
    return "\n".join(lines)


def answer(session_id: str, question: str) -> Dict:
    """
    Query the RAG pipeline.
    Returns: {answer, citations, chunks_used, context, images}
    """
    chunks = retrieve(session_id, question)

    if not chunks:
        return {
            "answer":      "No papers uploaded yet. Please upload a research paper first.",
            "citations":   [],
            "chunks_used": 0,
            "context":     "",
            "images":      [],
        }

    context = _format_context(chunks)
    user_msg = f"Paper excerpts:\n{context}\n\nQuestion: {question}"
    response = chat(SYSTEM_PROMPT, user_msg, temperature=0.1)

    # Build unique citations
    seen = set()
    citations = []
    for c in chunks:
        key = f"{c['filename']}|{c['section']}"
        if key not in seen:
            citations.append({
                "filename": c["filename"],
                "section":  c["section"],
                "pages":    c["pages"],
                "score":    c["score"],
                "type":     c.get("type", "text"),
            })
            seen.add(key)

    # Collect image paths from retrieved image chunks
    images = []
    for c in chunks:
        if c.get("type") == "image" and c.get("image_path"):
            images.append({
                "image_path": c["image_path"],
                "page":       c["pages"],
                "section":    c["section"],
            })

    return {
        "answer":      response,
        "citations":   citations,
        "chunks_used": len(chunks),
        "context":     context,
        "images":      images,
    }
