# Paper Element Retriever (Multimodal RAG)

Upload research papers or documents. Ask questions about **text AND figures**. Get cited, grounded answers.

> Powered by Ollama (local LLM + vision) + ChromaDB + sentence-transformers.  
> **Multimodal**: Extracts and understands figures, charts, and tables from PDFs.  

---

## Features

- **Text RAG** — section-aware PDF chunking (Abstract, Methods, Results, etc.)
- **Image RAG** — extracts figures/charts from PDFs, describes them using a vision model
- **Grounded answers** — LLM only uses retrieved context, never hallucinated knowledge
- **Citations** — every answer cites the section name and page number
- **6-metric evaluation** — built-in evaluation suite (faithfulness, relevancy, precision, recall, correctness)
- **Session-scoped** — each session is isolated, data auto-deleted on logout
- **Fully local** — no API keys, no cloud, everything runs on your machine

---

## Stack

| Layer | Technology |
|---|---|
| LLM (text) | Ollama — llama3.1:8b |
| LLM (vision) | Ollama — llava |
| Embeddings | sentence-transformers / all-MiniLM-L6-v2 |
| Vector store | ChromaDB (session-scoped, auto-deleted) |
| PDF parsing | PyMuPDF (section-aware chunking + image extraction) |
| Evaluation | 6-metric RAGAS-style suite (Ollama as judge) |
| API | FastAPI |
| UI | Vanilla HTML/CSS/JS |

---

## Installation

### Prerequisites

- **Python 3.10+**
- **Ollama** installed and running — [Download Ollama](https://ollama.com/download)

### Step 1: Clone the repository

```bash
git clone https://github.com/4M5/paper-element-retriever.git
```

### Step 2: Create a virtual environment (recommended)

Using conda:
```bash
conda create -n rag python=3.10 -y
conda activate rag
```

Or using venv:
```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows
```

### Step 3: Install Python dependencies

```bash
pip install -r requirements.txt
```

This installs:
| Package | Purpose |
|---|---|
| `chromadb` | Vector database for storing and searching embeddings |
| `sentence-transformers` | Converts text into 384-dimensional embedding vectors |
| `ollama` | Python client for local Ollama LLM |
| `pymupdf` | PDF text extraction and image extraction |
| `fastapi` | REST API framework |
| `uvicorn` | ASGI server to run FastAPI |
| `python-dotenv` | Loads config from `.env` file |
| `ragas`, `datasets` | RAG evaluation framework |
| `langchain*` | LLM orchestration (used by evaluation) |

### Step 4: Pull Ollama models

Make sure Ollama is installed and running, then pull both models:

```bash
# Text model (for answering questions) — ~4.7GB
ollama pull llama3.1:8b

# Vision model (for describing images/figures) — ~4.7GB
ollama pull llava
```

> **Note**: You only need to pull models once. They are stored locally by Ollama.

### Step 5: Configure environment

```bash
cp .env.example .env
```

The `.env` file contains all configurable settings:

| Variable | Default | What it does |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | LLM used for answering questions |
| `VISION_MODEL` | `llava` | Vision model used for describing images |
| `CHUNK_SIZE` | `600` | Number of words per text chunk |
| `CHUNK_OVERLAP` | `80` | Word overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of chunks retrieved per query |
| `MIN_IMAGE_SIZE` | `100` | Minimum image dimension (px) to process |

### Step 6: Run the application

```bash
python main.py
```

Open **http://localhost:8000** in your browser.

---

## Usage

1. Click **Start Session**
2. **Upload a PDF** (drag & drop or click the upload area)
   - Text is chunked by academic sections
   - Images are extracted and described by the vision model (takes 30-60s per image)
3. **Ask questions** about the document
   - *"What methodology did the authors use?"*
   - *"What does Figure 1 show?"*
   - *"Summarize the results"*
4. **Run Evaluation** (optional) — tests RAG quality with 6 metrics

---

## Project Structure

```
paper-element-retriever/
├── main.py                        # FastAPI server (multimodal)
├── requirements.txt               # Python dependencies
├── .env.example                   # Config template (copy to .env)
├── .gitignore
├── README.md
│
├── ingestion/                     # Document processing
│   ├── pdf_parser.py              # Section-aware PDF chunking + image extraction
│   ├── image_processor.py         # Vision model image descriptions
│   └── vector_store.py            # ChromaDB storage (text + image chunks)
│
├── query/                         # Answer generation
│   └── rag_engine.py              # Grounded LLM answering + citations + images
│
├── evaluation/                    # Quality testing
│   └── evaluator.py               # 6-metric evaluation suite
│
├── utils/                         # Shared utilities
│   ├── ollama_client.py           # Ollama wrapper (text + vision)
│   └── session_manager.py         # Session lifecycle management
│
└── frontend/                      # Web interface
    └── index.html                 # Single-page UI
```

---

## Evaluation Metrics

| Metric | What it checks | Pass threshold |
|---|---|---|
| Faithfulness | Answer only uses paper content | ≥ 0.80 |
| Answer Relevancy | Answer addresses the question | ≥ 0.75 |
| Context Precision | Best chunks ranked at top | ≥ 0.70 |
| Context Recall | Context covers the full answer | ≥ 0.70 |
| Context Relevancy | Retrieved chunks relate to query | ≥ 0.70 |
| Answer Correctness | Answer matches ground truth | ≥ 0.75 |

---

## Troubleshooting

| Problem | Solution |
|---|---|
| `Connection refused` on startup | Make sure Ollama is running: `ollama serve` |
| `Model not found` | Pull the model: `ollama pull llama3.1:8b` |
| Upload takes too long | Vision model needs ~30-60s per image (normal for local inference) |
| No images extracted | Check if your PDF has embedded images (not just scanned text) |
| Query hangs | Frontend has a 2-min timeout; check Ollama is responsive |
