"""
ingestion/vector_store.py

ChromaDB vector store — session-scoped.
Each session gets its own collection, deleted on logout/timeout.
Stores section metadata alongside each chunk for citation.
Supports both text chunks and image-description chunks.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import uuid, os
from dotenv import load_dotenv
import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

load_dotenv()

TOP_K    = int(os.getenv("TOP_K_RESULTS", 5))
_client  = chromadb.PersistentClient(path="./data/chroma")
_embedder = SentenceTransformer("all-MiniLM-L6-v2")


def _embed(texts: List[str]) -> List[List[float]]:
    return _embedder.encode(texts, show_progress_bar=False).tolist()


def ingest_chunks(session_id: str, filename: str, chunks: List[Dict]) -> int:
    """
    Store chunks in session-scoped ChromaDB collection.
    Each chunk carries: text, section, pages, filename, type, image_path.
    """
    collection = _client.get_or_create_collection(
        name=f"sess_{session_id}",
        metadata={"session_id": session_id},
    )

    texts      = [c["text"] for c in chunks]
    embeddings = _embed(texts)
    ids        = [str(uuid.uuid4()) for _ in chunks]
    metadatas  = [
        {
            "filename":   filename,
            "section":    c.get("section", "Unknown"),
            "pages":      str(c.get("pages", [])),
            "type":       c.get("type", "text"),
            "image_path": c.get("image_path", ""),
        }
        for c in chunks
    ]

    collection.add(
        documents=texts,
        embeddings=embeddings,
        ids=ids,
        metadatas=metadatas,
    )

    text_count  = sum(1 for c in chunks if c.get("type", "text") == "text")
    image_count = sum(1 for c in chunks if c.get("type") == "image")
    print(f"[VectorStore] Stored {text_count} text + {image_count} image chunks "
          f"for session '{session_id}'")
    return len(chunks)


def retrieve(session_id: str, query: str, top_k: int = None) -> List[Dict]:
    """
    Retrieve top-k relevant chunks for a query.
    Returns [{text, section, pages, filename, score, type, image_path}]
    """
    k = top_k or TOP_K
    try:
        collection = _client.get_collection(name=f"sess_{session_id}")
    except Exception:
        return []

    if collection.count() == 0:
        return []

    results = collection.query(
        query_embeddings=_embed([query]),
        n_results=min(k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    output = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ):
        output.append({
            "text":       doc,
            "section":    meta.get("section", "Unknown"),
            "pages":      meta.get("pages", "[]"),
            "filename":   meta.get("filename", "Unknown"),
            "score":      round(1 - dist, 3),
            "type":       meta.get("type", "text"),
            "image_path": meta.get("image_path", ""),
        })
    return output


def delete_session(session_id: str) -> bool:
    try:
        _client.delete_collection(name=f"sess_{session_id}")
        print(f"[VectorStore] Deleted session '{session_id}'")
        return True
    except Exception:
        return False


def list_papers(session_id: str) -> List[str]:
    try:
        collection = _client.get_collection(name=f"sess_{session_id}")
        results    = collection.get(include=["metadatas"])
        return sorted({m["filename"] for m in results["metadatas"]})
    except Exception:
        return []


def get_all_chunks(session_id: str) -> List[Dict]:
    """Used by evaluation pipeline."""
    try:
        collection = _client.get_collection(name=f"sess_{session_id}")
        results    = collection.get(include=["documents", "metadatas"])
        return [
            {
                "text":       doc,
                "section":    meta.get("section"),
                "filename":   meta.get("filename"),
                "type":       meta.get("type", "text"),
                "image_path": meta.get("image_path", ""),
            }
            for doc, meta in zip(results["documents"], results["metadatas"])
        ]
    except Exception:
        return []
