# Multimodal RAG — How It Works

A technical guide to how this system processes PDFs, stores data, retrieves context, and generates answers.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  User uploads PDF                                               │
│     ↓                                                           │
│  ┌──────────────────────┐    ┌─────────────────────────────┐    │
│  │  pdf_parser.py        │    │  pdf_parser.py               │    │
│  │  parse_paper()        │    │  extract_images()            │    │
│  │  PyMuPDF (fitz)       │    │  PyMuPDF (fitz)              │    │
│  │     ↓                 │    │     ↓                        │    │
│  │  Text chunks          │    │  PNG images saved to disk    │    │
│  │  {text, section,      │    │  {path, page, width, height} │    │
│  │   pages, type:"text"} │    │     ↓                        │    │
│  └──────────┬───────────┘    │  image_processor.py           │    │
│              │                │  process_images()             │    │
│              │                │  Ollama llava vision model    │    │
│              │                │     ↓                        │    │
│              │                │  Image description chunks    │    │
│              │                │  {text, section,             │    │
│              │                │   pages, type:"image",       │    │
│              │                │   image_path}                │    │
│              │                └──────────┬──────────────────┘    │
│              │                           │                       │
│              └───────────┬───────────────┘                       │
│                          ↓                                       │
│              ┌──────────────────────┐                            │
│              │  vector_store.py      │                            │
│              │  ingest_chunks()      │                            │
│              │  SentenceTransformer  │                            │
│              │  all-MiniLM-L6-v2     │                            │
│              │     ↓                 │                            │
│              │  ChromaDB collection  │                            │
│              └──────────────────────┘                            │
│                                                                  │
│  User asks a question                                            │
│     ↓                                                           │
│  ┌──────────────────────┐    ┌─────────────────────────────┐    │
│  │  vector_store.py      │    │  rag_engine.py               │    │
│  │  retrieve()           │    │  answer()                    │    │
│  │  embed query →        │    │  format context →            │    │
│  │  cosine similarity    │ →  │  send to Ollama llama3.1:8b  │    │
│  │  return top-k chunks  │    │  return answer + citations   │    │
│  └──────────────────────┘    │  + referenced images          │    │
│                               └─────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Step-by-Step Flow

### 1. PDF Text Extraction

**File**: `ingestion/pdf_parser.py`  
**Library**: `PyMuPDF` (imported as `fitz`)

| Function | What it does |
|---|---|
| `_extract_raw_text(pdf_path)` | Opens PDF with `fitz.open()`, calls `page.get_text()` for each page. Returns `[{page: 1, text: "..."}]` |
| `_is_section_header(line)` | Regex-matches lines like "Abstract", "2. Methods", "Results" to detect academic sections |
| `_split_into_sections(pages)` | Walks line-by-line, groups text by detected section headers. Returns `[{section: "Methods", text: "...", pages: [3,4]}]` |
| `_chunk_section(section)` | Splits long sections into overlapping word-based chunks (default 600 words, 80 overlap) |
| `parse_paper(pdf_path)` | **Main pipeline**: raw text → sections → chunks. Returns list of text chunks |

**Each text chunk looks like:**
```python
{
    "text":    "The patients were divided into two groups...",
    "section": "Methods",
    "pages":   [3, 4],
    "type":    "text"       # marks this as a text chunk
}
```

### 2. Image Extraction

**File**: `ingestion/pdf_parser.py`  
**Library**: `PyMuPDF` (`fitz`)

| Function | What it does |
|---|---|
| `extract_images(pdf_path, session_id, filename)` | For each page: calls `page.get_images(full=True)` to get embedded image refs (xrefs), creates a `fitz.Pixmap` for each, converts CMYK→RGB if needed, saves as PNG to `./uploads/{session_id}/images/` |

**Output:**
```python
{
    "path":        "./uploads/abc123/images/paper_p3_img1.png",
    "page":        3,
    "width":       750,
    "height":      500,
    "image_index": 1
}
```

### 3. Image → Text Description (Vision Model)

**File**: `ingestion/image_processor.py`  
**Library**: `ollama` (Python client)

| Function | What it does |
|---|---|
| `_is_meaningful_image(w, h)` | Filters out images smaller than 100×100px or extremely narrow (decorative borders) |
| `process_single_image(path, ...)` | Calls `describe_image()` from `ollama_client.py` — sends the image to llava and gets a text description back |
| `process_images(images, ...)` | Batch-processes all meaningful images, returns description chunks |

**File**: `utils/ollama_client.py`

| Function | What it does |
|---|---|
| `describe_image(image_path)` | Reads image as base64, sends to Ollama's vision model with a research-focused prompt, returns text description |

The prompt tells llava:
> *"Describe this image in detail: what type of figure is it (chart, diagram, photo, table), what does it show, what are the key data points, labels, axes, or trends?"*

**Each image chunk looks like:**
```python
{
    "text":       "[Figure from page 3] This is a bar chart showing patient outcomes...",
    "section":    "Figure (page 3)",
    "pages":      [3],
    "type":       "image",         # marks this as an image chunk
    "image_path": "./uploads/abc123/images/paper_p3_img1.png"
}
```

### 4. Embedding & Storage (Vector Store)

**File**: `ingestion/vector_store.py`  
**Libraries**: `sentence-transformers`, `chromadb`

| Function | What it does |
|---|---|
| `_embed(texts)` | Uses `SentenceTransformer("all-MiniLM-L6-v2")` to convert text → 384-dim float vectors |
| `ingest_chunks(session_id, filename, chunks)` | Embeds all chunks (text + image descriptions), stores in a ChromaDB collection named `sess_{session_id}` |

**How ChromaDB stores each chunk:**

```
Collection: sess_abc123
┌─────────────────────────────────────────────────────────────┐
│  id:        "uuid-1234"                                      │
│  document:  "The patients were divided..."  (searchable text)│
│  embedding: [0.12, -0.34, 0.56, ...]       (384-dim vector) │
│  metadata:                                                   │
│    filename:   "paper.pdf"                                   │
│    section:    "Methods"                                      │
│    pages:      "[3, 4]"                                      │
│    type:       "text"                (or "image")            │
│    image_path: ""                    (or path to PNG)        │
└─────────────────────────────────────────────────────────────┘
```

**Key insight**: Image descriptions are embedded as regular text using the *same* text embedding model. This means "what does the chart show?" can match against an image description like "This is a bar chart showing patient outcomes..." via cosine similarity.

### 5. Retrieval (Query Time)

**File**: `ingestion/vector_store.py`

| Function | What it does |
|---|---|
| `retrieve(session_id, query, top_k=5)` | Embeds the query, calls `collection.query()` with cosine distance, returns top-k most similar chunks (both text and image descriptions) |

**Similarity search:**
```
Query: "What does the chart show?"
    ↓ embed → [0.45, -0.12, ...]
    ↓ cosine similarity against all stored vectors
    ↓ return top 5 closest matches

Result: might include both text chunks AND image description chunks
```

### 6. Answer Generation

**File**: `query/rag_engine.py`  
**Library**: `ollama`

| Function | What it does |
|---|---|
| `_format_context(chunks)` | Formats retrieved chunks as labeled excerpts. Text chunks show `[Excerpt N | Section: X | Pages: Y]`. Image chunks show `[Image/Figure N | Page: Y]` |
| `answer(session_id, question)` | Formats context → sends to Ollama `llama3.1:8b` → returns answer + citations + referenced image paths |

---

## Libraries Used

| Library | Purpose | Where Used |
|---|---|---|
| `PyMuPDF` (fitz) | PDF text extraction + image extraction | `pdf_parser.py` |
| `sentence-transformers` | Text → 384-dim embedding vectors | `vector_store.py` |
| `chromadb` | Vector database (stores & searches embeddings) | `vector_store.py` |
| `ollama` | Local LLM (text generation + vision) | `ollama_client.py` |
| `FastAPI` | REST API server | `main.py` |
| `uvicorn` | ASGI server to run FastAPI | `main.py` |
| `python-dotenv` | Load `.env` config | everywhere |

---

## Data Flow Summary

```
PDF Upload
    │
    ├─► parse_paper()        → text chunks     ─┐
    │   [PyMuPDF]                                │
    │                                            ├─► ingest_chunks()
    └─► extract_images()     → PNG files         │   [sentence-transformers
        [PyMuPDF]                │               │    + ChromaDB]
                                 │               │
                    process_images()             ─┘
                    [Ollama llava]
                    → image description chunks

Query
    │
    ├─► retrieve()           → top-k chunks (text + images)
    │   [sentence-transformers + ChromaDB]
    │
    └─► answer()             → LLM response + citations + images
        [Ollama llama3.1:8b]
```
