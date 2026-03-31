"""
main.py — Research Paper RAG API (Multimodal)
"""

import os, shutil, tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

from ingestion.pdf_parser import parse_paper, extract_metadata, extract_images
from ingestion.vector_store import ingest_chunks, list_papers
from ingestion.image_processor import process_images
from query.rag_engine import answer
from evaluation.evaluator import RAGEvaluator
from utils.session_manager import create_session, end_session, touch, add_paper, get_info
from utils.ollama_client import check_model, check_vision_model

CHUNK_SIZE    = int(os.getenv("CHUNK_SIZE", 600))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 80))
FRONTEND      = Path(__file__).parent / "frontend" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    Path("./data").mkdir(exist_ok=True)
    Path("./uploads").mkdir(exist_ok=True)
    check_model()
    check_vision_model()
    yield
    # shutdown (nothing needed)


app = FastAPI(title="Research Paper RAG (Multimodal)", version="2.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Mount uploads directory to serve extracted images
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")


@app.get("/", response_class=HTMLResponse)
def ui():
    return HTMLResponse(FRONTEND.read_text() if FRONTEND.exists() else "<h2>Research RAG running — see /docs</h2>")


# ── Session ───────────────────────────────────────────────────────────────

@app.post("/session/start")
def start():
    sid = create_session()
    return {"session_id": sid}

@app.post("/session/end")
def stop(session_id: str = Form(...)):
    if not get_info(session_id):
        raise HTTPException(404, "Session not found")
    # Also clean up extracted images
    image_dir = Path(f"./uploads/{session_id}")
    if image_dir.exists():
        shutil.rmtree(image_dir)
    end_session(session_id)
    return {"message": "Session ended. All data deleted."}


# ── Upload ────────────────────────────────────────────────────────────────

@app.post("/upload")
async def upload(session_id: str = Form(...), file: UploadFile = File(...)):
    if not get_info(session_id):
        raise HTTPException(404, "Start a session first.")
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(400, "Only PDF files are supported.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        meta   = extract_metadata(tmp_path)

        # ── Text chunks ──
        chunks = parse_paper(tmp_path, CHUNK_SIZE, CHUNK_OVERLAP)

        # ── Image extraction + vision processing ──
        raw_images = extract_images(tmp_path, session_id, file.filename)
        image_chunks = []
        if raw_images:
            try:
                image_chunks = process_images(raw_images, session_id, file.filename)
            except Exception as e:
                print(f"[Upload] Image processing failed (continuing with text only): {e}")

        # ── Reject only if BOTH text and images are empty ──
        if not chunks and not image_chunks:
            raise HTTPException(422, "Could not extract any content (text or images) from this PDF.")

        # ── Ingest all chunks (text + image descriptions) ──
        all_chunks = chunks + image_chunks
        stored = ingest_chunks(session_id, file.filename, all_chunks)
        add_paper(session_id)
        touch(session_id)

        return {
            "message":      f"'{file.filename}' ingested successfully.",
            "metadata":     meta,
            "text_chunks":  len(chunks),
            "image_chunks": len(image_chunks),
            "total_chunks": stored,
            "images_found": len(raw_images),
            "sections":     list({c["section"] for c in chunks}),
        }
    finally:
        os.unlink(tmp_path)


# ── Query ─────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    session_id: str
    question:   str

@app.post("/query")
def query(req: QueryRequest):
    if not get_info(req.session_id):
        raise HTTPException(404, "Session not found.")
    touch(req.session_id)
    result = answer(req.session_id, req.question)
    return result


# ── Image serving ─────────────────────────────────────────────────────────

@app.get("/images/{session_id}/{filename}")
def get_image(session_id: str, filename: str):
    """Serve an extracted image by session and filename."""
    image_path = Path(f"./uploads/{session_id}/images/{filename}")
    if not image_path.exists():
        raise HTTPException(404, "Image not found.")
    return FileResponse(str(image_path), media_type="image/png")


# ── Papers list ───────────────────────────────────────────────────────────

@app.get("/papers/{session_id}")
def papers(session_id: str):
    return {"papers": list_papers(session_id)}


# ── Evaluation ────────────────────────────────────────────────────────────

@app.post("/evaluate")
def evaluate(session_id: str = Form(...), n: int = Form(10)):
    if not get_info(session_id):
        raise HTTPException(404, "Session not found.")
    evaluator = RAGEvaluator(session_id)
    report    = evaluator.run(n=n)
    path      = f"./data/eval_{session_id}.json"
    evaluator.save_report(report, path)
    return {
        "n_test_cases":    report["n_test_cases"],
        "avg_latency_sec": report["avg_latency_sec"],
        "metrics":         {k: v["mean"] for k, v in report["metrics"].items()},
    }


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "model":        os.getenv("OLLAMA_MODEL", "llama3.2"),
        "vision_model": os.getenv("VISION_MODEL", "llava"),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
