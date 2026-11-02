# Paper Element Retriever

**A Traditional Multimodal Retrieval-Augmented Generation (RAG) system for understanding and analysing research papers.**

Paper Element Retriever helps users explore lengthy academic papers without repeatedly searching through the entire document. It processes PDF text and embedded figures, retrieves relevant information, and generates answers grounded in the uploaded paper with section and page references.

The project was initially built during my internship to address a practical challenge: reading and analysing long research papers is time-consuming, and answers from a general-purpose LLM may contain information that is not supported by the paper. I explored RAG as a way to make paper analysis more efficient and reduce the risk of unsupported or hallucinated responses.

> **Note:** RAG can help ground answers in source material, but it does not guarantee that hallucinations are eliminated. Always verify important claims against the cited paper.

![Paper Element Retriever](assets/overview.png)

## What it does

Upload a research paper, ask questions about it, and receive an answer based on retrieved passages and figure descriptions.

Example questions:

- What problem does this paper aim to solve?
- Which methodology or model did the authors use?
- What does Figure 1 illustrate?
- What are the main findings?
- Which limitations do the authors identify?

## How it works

The current implementation is a **traditional dense-retrieval RAG pipeline with multimodal document ingestion**. It processes both textual and visual content, converts visual content into text descriptions, and retrieves the resulting text representations.

1. **PDF ingestion:** Extracts document text, metadata, page information, and embedded images.
2. **Section-aware chunking:** Splits extracted text into chunks while retaining section and page metadata.
3. **Visual processing:** Sends eligible extracted figures to a vision model (LLaVA) to generate textual descriptions.
4. **Embedding and indexing:** Encodes text chunks and figure descriptions with Sentence Transformers and stores them in ChromaDB.
5. **Retrieval:** Uses dense vector similarity to retrieve relevant text chunks and figure descriptions for a user query.
6. **Grounded generation:** Passes the retrieved context to a local language model (Llama 3.1) to generate an answer with source references.

### Architecture

```text
                         Research Paper (PDF)
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
             Text Extraction              Image Extraction
                    |                           |
                    v                           v
          Section-aware Chunking          Vision Model (LLaVA)
                    |                           |
                    v                           v
              Text Chunks               Figure Descriptions
                    |                           |
                    +-------------+-------------+
                                  |
                                  v
                         Sentence Embeddings
                                  |
                                  v
                              ChromaDB
                                  |
                             User Query
                                  |
                                  v
                         Dense Vector Search
                                  |
                                  v
                       Retrieved Text Context
                                  |
                                  v
                        Local LLM (Llama 3.1)
                                  |
                                  v
                    Answer with Section/Page References
```

**Multimodal scope:** The system handles text and images during ingestion, but the current retrieval index stores text chunks and vision-generated figure descriptions. It does not perform direct image-embedding retrieval or visual search over image vectors.

## Features

### Research paper understanding
- Extracts text from PDF documents.
- Uses section-aware chunking to preserve document structure.
- Retains page and section metadata for source references.

### Figure understanding
- Extracts eligible embedded images from PDFs.
- Uses a local vision model to produce textual descriptions.
- Makes figure-derived information available to the text retrieval pipeline.

### Retrieval-grounded question answering
- Retrieves relevant chunks from the uploaded paper.
- Uses retrieved context to guide answer generation.
- Returns section and page references to help users verify information.

### Local inference
- Uses Ollama to run the language and vision models locally.
- Does not require an external LLM API for the main question-answering flow.

### Evaluation
The project includes an exploratory evaluation pipeline covering:
- Faithfulness
- Answer relevancy
- Context precision
- Context recall
- Context relevancy
- Answer correctness
- Average query latency

The current evaluation uses automatically generated question-answer cases and LLM-based judging. Treat these scores as development indicators, not as a substitute for a fixed, human-verified benchmark.

### Session handling
- Maintains separate retrieval contexts for sessions.
- Supports ending a session and cleaning up its associated data.

## Technology Stack

| Component | Technology |
|---|---|
| Programming Language | Python |
| API | FastAPI |
| Text LLM | Ollama – Llama 3.1 8B |
| Vision Model | Ollama – LLaVA |
| Embeddings | Sentence Transformers |
| Vector Database | ChromaDB |
| PDF Processing | PyMuPDF |
| Frontend | HTML, CSS, JavaScript |
| Evaluation | Custom LLM-as-judge evaluation pipeline |

## Project Structure

```text
paper-element-retriever/
├── main.py
├── requirements.txt
├── .env.example
├── README.md
├── ingestion/
│   ├── pdf_parser.py
│   ├── image_processor.py
│   └── vector_store.py
├── query/
│   └── rag_engine.py
├── evaluation/
│   └── evaluator.py
├── utils/
│   ├── ollama_client.py
│   └── session_manager.py
└── frontend/
    └── index.html
```

## Getting Started

### Prerequisites

- Python 3.10 or later
- [Ollama](https://ollama.com/) installed and running
- Enough system memory and storage for the selected local models

### 1. Clone the repository

```bash
git clone https://github.com/4M5/paper-element-retriever.git
cd paper-element-retriever
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
```

Windows:

```bash
venv\\Scripts\\activate
```

Linux/macOS:

```bash
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Download the Ollama models

Make sure the Ollama service is running, then run:

```bash
ollama pull llama3.1:8b
ollama pull llava
```

### 5. Configure environment variables

Copy `.env.example` to `.env` and adjust the values if needed.

Common configuration options:

| Variable | Default | Purpose |
|---|---:|---|
| `OLLAMA_MODEL` | `llama3.1:8b` | Text generation model |
| `VISION_MODEL` | `llava` | Figure description model |
| `CHUNK_SIZE` | `600` | Text chunk size |
| `CHUNK_OVERLAP` | `80` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Number of retrieved results |
| `MIN_IMAGE_SIZE` | `100` | Minimum image dimension for processing |

### 6. Run the application

```bash
python main.py
```

Open [http://localhost:8000](http://localhost:8000) in your browser.

## Usage

1. Start a session in the application.
2. Upload a research paper in PDF format.
3. Wait for text and figure processing to complete.
4. Ask a question about the uploaded paper.
5. Review the generated answer and its section/page references.
6. Run the evaluation option to inspect the available evaluation results.
7. End the session when finished.

## Current Limitations

- Scanned PDFs and image-only pages are not handled with a dedicated OCR pipeline.
- Only eligible embedded PDF images are extracted; some vector graphics and composite figures may not be captured as complete figures.
- Figure descriptions depend on the vision model and may omit details from complex plots, diagrams, or tables.
- Retrieval currently uses dense vector similarity; keyword-based hybrid retrieval and reranking are not implemented.
- The current pipeline does not directly index image embeddings.
- Citation references are useful for navigation, but they should be checked against the original paper.
- The evaluation pipeline is exploratory and does not yet use a fixed, human-verified benchmark.
- This is a research and portfolio prototype, not a production-ready document service.

## Roadmap

The next stage is to evolve the baseline into a more capable research-paper assistant while preserving the current implementation for comparison.

- [ ] **Advanced retrieval:** Add BM25 keyword search alongside dense vector search.
- [ ] **Result fusion:** Combine retrieval rankings using Reciprocal Rank Fusion (RRF).
- [ ] **Reranking:** Add a cross-encoder reranker to improve the relevance of retrieved passages.
- [ ] **Query enhancement:** Explore query rewriting and decomposition for complex questions.
- [ ] **Better document parsing:** Add OCR and improve extraction of tables, equations, and page layout.
- [ ] **Richer multimodal retrieval:** Explore direct visual embeddings and page-level visual retrieval.
- [ ] **Evidence-grounded answers:** Validate citations against retrieved chunks and support claim-level evidence.
- [ ] **Reliable evaluation:** Build a fixed, human-reviewed dataset and report retrieval metrics such as Recall@K, MRR, and nDCG alongside answer quality and latency.
- [ ] **Multi-paper analysis:** Support comparison and evidence synthesis across multiple research papers.

## Project Motivation

Research papers often contain valuable information spread across many pages, sections, figures, and tables. Finding a specific method, result, or explanation can take considerable time. This project was an exploration of how retrieval-augmented generation and vision-language models can make that process more accessible while keeping answers connected to the source document.

The current version establishes a traditional multimodal RAG baseline. The roadmap focuses on improving retrieval quality, multimodal understanding, evidence traceability, and evaluation.

---

**Built with Python, ChromaDB, Sentence Transformers, FastAPI, and Ollama.**
