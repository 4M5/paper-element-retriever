"""
ingestion/pdf_parser.py

Parses research papers / journal PDFs.
Splits by academic sections: Abstract, Introduction, Methods,
Results, Discussion, Conclusion, References.

Also extracts images/figures for multimodal RAG.

Why section-aware chunking?
  Plain chunking splits mid-sentence across sections.
  Section-aware chunking keeps context intact — a question about
  "methodology" retrieves from the Methods section, not random chunks.
"""

import fitz  # PyMuPDF
import re, os
from typing import List, Dict, Tuple
from pathlib import Path

# Academic section headers to detect
SECTION_PATTERNS = [
    r"^abstract$",
    r"^introduction$",
    r"^(related work|background|literature review)$",
    r"^(methodology|methods|materials and methods|experimental setup)$",
    r"^(results|findings|experiments)$",
    r"^(discussion|analysis)$",
    r"^(conclusion|conclusions|concluding remarks)$",
    r"^(references|bibliography)$",
    r"^(appendix|supplementary).*$",
    r"^\d+\.?\s+(introduction|method|result|discussion|conclusion).*$",
]


def _is_section_header(line: str) -> bool:
    clean = line.strip().lower()
    for pattern in SECTION_PATTERNS:
        if re.match(pattern, clean):
            return True
    # Detect numbered sections like "2. Methodology" or "3.1 Results"
    if re.match(r"^\d+(\.\d+)?\s+[A-Z]", line.strip()):
        return True
    return False


def _extract_raw_text(pdf_path: str) -> List[Dict]:
    """Extract text page by page, preserving page numbers."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text()
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    doc.close()
    return pages


def _split_into_sections(pages: List[Dict]) -> List[Dict]:
    """
    Split full text into named sections.
    Returns list of {section, text, pages}
    """
    full_lines = []
    for page in pages:
        for line in page["text"].split("\n"):
            full_lines.append({"line": line, "page": page["page"]})

    sections = []
    current_section = "header"
    current_lines = []
    current_pages = set()

    for item in full_lines:
        line = item["line"]
        page = item["page"]

        if _is_section_header(line) and len(line.strip()) < 80:
            if current_lines:
                sections.append({
                    "section": current_section,
                    "text": "\n".join(current_lines).strip(),
                    "pages": sorted(current_pages),
                })
            current_section = line.strip()
            current_lines = []
            current_pages = set()
        else:
            if line.strip():
                current_lines.append(line)
                current_pages.add(page)

    # Last section
    if current_lines:
        sections.append({
            "section": current_section,
            "text": "\n".join(current_lines).strip(),
            "pages": sorted(current_pages),
        })

    return [s for s in sections if len(s["text"]) > 100]


def _chunk_section(section: Dict, chunk_size: int = 600, overlap: int = 80) -> List[Dict]:
    """
    Split a section's text into overlapping chunks.
    Each chunk carries its section name and page numbers.
    """
    text = section["text"]
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end])
        chunks.append({
            "text":    chunk_text,
            "section": section["section"],
            "pages":   section["pages"],
            "type":    "text",
        })
        if end == len(words):
            break
        start += chunk_size - overlap

    return chunks


def parse_paper(pdf_path: str, chunk_size: int = 600, overlap: int = 80) -> List[Dict]:
    """
    Full pipeline: PDF → sections → chunks.

    Returns list of chunks, each with:
        text    : chunk content
        section : section name (e.g. "Methods")
        pages   : list of page numbers this chunk spans
        type    : "text" (image chunks added separately)
    """
    pages    = _extract_raw_text(pdf_path)
    sections = _split_into_sections(pages)

    all_chunks = []
    for section in sections:
        chunks = _chunk_section(section, chunk_size, overlap)
        all_chunks.extend(chunks)

    print(f"[Parser] {len(sections)} sections → {len(all_chunks)} text chunks")
    return all_chunks


def extract_images(pdf_path: str, session_id: str, filename: str) -> List[Dict]:
    """
    Extract images from a PDF and save them to disk.

    Returns list of image metadata:
        {path, page, width, height, image_index}
    """
    doc = fitz.open(pdf_path)

    # Create output directory for this session's images
    image_dir = Path(f"./uploads/{session_id}/images")
    image_dir.mkdir(parents=True, exist_ok=True)

    images = []
    image_count = 0

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        image_list = page.get_images(full=True)

        for img_idx, img in enumerate(image_list):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)

                # Convert CMYK to RGB if needed
                if pix.n > 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)

                width, height = pix.width, pix.height
                image_count += 1

                # Save image as PNG
                img_filename = f"{Path(filename).stem}_p{page_idx + 1}_img{img_idx + 1}.png"
                img_path = str(image_dir / img_filename)
                pix.save(img_path)

                images.append({
                    "path":        img_path,
                    "page":        page_idx + 1,
                    "width":       width,
                    "height":      height,
                    "image_index": image_count,
                })

            except Exception as e:
                print(f"[Parser] Could not extract image {img_idx} from page {page_idx + 1}: {e}")
                continue

    doc.close()
    print(f"[Parser] Extracted {len(images)} images from '{filename}'")
    return images


def extract_metadata(pdf_path: str) -> Dict:
    """
    Extract paper title, authors, year from first page.
    Best-effort — research PDFs have no standard format.
    """
    doc  = fitz.open(pdf_path)
    meta = doc.metadata
    first_page_text = doc[0].get_text() if len(doc) > 0 else ""
    doc.close()

    lines = [l.strip() for l in first_page_text.split("\n") if l.strip()]

    return {
        "title":      meta.get("title") or (lines[0] if lines else "Unknown"),
        "author":     meta.get("author", "Unknown"),
        "year":       meta.get("creationDate", "")[:4] if meta.get("creationDate") else "Unknown",
        "page_count": len(fitz.open(pdf_path)),
    }

