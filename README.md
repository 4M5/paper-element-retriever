# Paper Element Retriever

A multimodal Retrieval-Augmented Generation (RAG) system for asking questions about research papers using both text and figures.

The system extracts section-aware text and figures from PDFs, processes them locally, retrieves relevant content, and generates cited answers using local language and vision models.

![Paper Element Retriever](assets/overview.png)

## Overview

Traditional RAG systems often focus only on text. This project extends the retrieval pipeline to include visual elements such as figures, charts, and tables found in research papers.

The pipeline is:

PDF
→ Text & Image Extraction
→ Section-aware Chunking
→ Embedding & Vector Storage
→ Multimodal Retrieval
→ Local LLM
→ Cited Answer

## Key Features

### Multimodal Retrieval

- Extracts text from research papers
- Splits text using academic section structure
- Extracts figures and other embedded images
- Uses a vision model to generate descriptions for extracted images
- Stores text and image-derived content for retrieval

### Grounded Question Answering

Questions are answered using retrieved document context rather than relying solely on the language model's general knowledge.

Answers include document references such as section names and page numbers.

Example questions:

- What methodology did the authors use?
- What does Figure 1 show?
- What were the main results?
- What limitations did the authors mention?

### Local LLM and Vision Models

The system uses Ollama for local inference:

- `llama3.1:8b` for text generation
- `llava` for image understanding

No external LLM API is required.

### Evaluation

The project includes an evaluation pipeline covering six RAG quality metrics:

- Faithfulness
- Answer Relevancy
- Context Precision
- Context Recall
- Context Relevancy
- Answer Correctness

Evaluation reports also include average query latency.

### Session Isolation

Each user session has its own retrieval context.

Uploaded and extracted data can be removed when the session ends, preventing data from being shared across sessions.

## Architecture

```text
                    ┌─────────────────┐
                    │   Research PDF  │
                    └────────┬────────┘
                             │
                 ┌───────────┴───────────┐
                 │                       │
                 ▼                       ▼
          Text Extraction          Image Extraction
                 │                       │
                 ▼                       ▼
        Section-aware Chunks      Vision Processing
                 │                       │
                 └───────────┬───────────┘
                             ▼
                     Vector Storage
                        ChromaDB
                             │
                             ▼
                       User Query
                             │
                             ▼
                    Retrieval Pipeline
                             │
                             ▼
                    Local LLM (Ollama)
                             │
                             ▼
                    Cited Answer
```

## Technology Stack

| Component | Technology |
|---|---|
| Language | Python |
| LLM | Ollama + Llama 3.1 8B |
| Vision Model | Ollama + LLaVA |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| API | FastAPI |
| Frontend | HTML, CSS, JavaScript |
| Evaluation | RAGAS-style evaluation pipeline |

## Project Structure

```text
paper-element-retriever/
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── ingestion/
│   ├── __init__.py
│   ├── pdf_parser.py
│   ├── image_processor.py
│   └── vector_store.py
│
├── query/
│   ├── __init__.py
│   └── rag_engine.py
│
├── evaluation/
│   ├── __init__.py
│   └── evaluator.py
│
├── utils/
│   ├── __init__.py
│   ├── ollama_client.py
│   └── session_manager.py
│
└── frontend/
    └── index.html
```

## Installation

### Prerequisites

- Python 3.10+
- Ollama installed and running
- Sufficient RAM/storage for the selected local models

### 1. Clone the repository

```bash
git clone https://github.com/4M5/paper-element-retriever.git
cd paper-element-retriever
```

### 2. Create a virtual environment

Using `venv`:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Install Ollama models

Make sure Ollama is installed and running, then pull the required models:

```bash
ollama pull llama3.1:8b
ollama pull llava
```

### 5. Configure environment variables

Copy `.env.example` to `.env`:

```bash
cp .env.example .env
```

On Windows, you can also copy the file manually.

The main configuration options include:

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Text generation model |
| `VISION_MODEL` | `llava` | Vision model for extracted images |
| `CHUNK_SIZE` | `600` | Text chunk size |
| `CHUNK_OVERLAP` | `80` | Chunk overlap |
| `TOP_K_RESULTS` | `5` | Number of retrieved results |
| `MIN_IMAGE_SIZE` | `100` | Minimum image dimension for processing |

### 6. Run the application

```bash
python main.py
```

Open the application in your browser:

```text
http://localhost:8000
```

## Usage

1. Start a new session.
2. Upload a research paper in PDF format.
3. The system extracts text and embedded images.
4. Text is divided into section-aware chunks.
5. Extracted figures are processed using the vision model.
6. Text and image-derived content are stored in ChromaDB.
7. Ask questions about the uploaded paper.
8. The system retrieves relevant context and generates a cited answer.
9. Optionally run the evaluation pipeline.

### Example Questions

- What methodology did the authors use?
- What does Figure 1 show?
- What were the main results?
- What limitations did the authors mention?

## Evaluation

The project includes an evaluation pipeline for measuring different aspects of RAG performance.

| Metric | What it checks |
|---|---|
| Faithfulness | Whether the generated answer is supported by the retrieved context |
| Answer Relevancy | Whether the answer addresses the user's question |
| Context Precision | Whether relevant retrieved chunks are ranked highly |
| Context Recall | Whether sufficient information was retrieved |
| Context Relevancy | Whether the retrieved context is relevant to the query |
| Answer Correctness | How closely the answer matches the expected answer |

The evaluation pipeline also records average query latency.

## Limitations

- Local LLM and vision inference can be computationally expensive.
- Processing figures can increase document ingestion time.
- Results depend on the quality of PDF text and image extraction.
- Scanned PDFs may require additional OCR processing.
- Vision model descriptions may not perfectly capture complex charts or diagrams.
- Retrieval quality depends on chunking, embeddings, and the selected `TOP_K_RESULTS`.
- The current implementation is a research/portfolio prototype rather than a production-scale RAG service.

## Future Improvements

- OCR support for scanned research papers
- Hybrid keyword and vector retrieval
- Reranking of retrieved chunks
- Improved table extraction
- Multi-document question answering
- Streaming responses
- Persistent production storage
- Larger and more diverse evaluation datasets

## Project Context

This project explores multimodal Retrieval-Augmented Generation for research-paper question answering, with a focus on document structure, visual information retrieval, local LLM inference, citation grounding, and measurable RAG evaluation.
