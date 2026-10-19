"""
src/mcp_server/rag/chunker.py
─────────────────────────────
Splits documents into overlapping text chunks for RAG indexing.
Supports Markdown/plain-text and PDF (via pymupdf).
Each chunk carries metadata (source file, chunk index, heading/page).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Chunk:
    text: str
    source: str          # relative path from docs root
    chunk_index: int
    heading: str = ""    # nearest heading (markdown) or "Page N" (PDF)


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Split *text* into overlapping chunks of approximately *chunk_size* chars.
    Tries to break at paragraph boundaries first, then falls back to hard cut.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").strip()
    paragraphs = re.split(r"\n{2,}", text)

    chunks: list[Chunk] = []
    current = ""
    current_heading = ""
    chunk_idx = 0

    for para in paragraphs:
        # Track the nearest heading
        heading_match = re.match(r"^#{1,3}\s+(.+)", para.strip())
        if heading_match:
            current_heading = heading_match.group(1).strip()

        if len(current) + len(para) + 2 > chunk_size and current:
            chunks.append(Chunk(
                text=current.strip(),
                source=source,
                chunk_index=chunk_idx,
                heading=current_heading,
            ))
            chunk_idx += 1
            # Overlap: keep last *overlap* chars
            current = current[-overlap:] + "\n\n" + para if overlap else para
        else:
            current = (current + "\n\n" + para).lstrip()

    if current.strip():
        chunks.append(Chunk(
            text=current.strip(),
            source=source,
            chunk_index=chunk_idx,
            heading=current_heading,
        ))

    return chunks


def chunk_pdf(
    path: Path,
    docs_root: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Extract text from a PDF with pymupdf and return overlapping chunks.
    Each chunk's heading is set to "Page N" so retrieval results show
    which page the passage came from.
    """
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF indexing. Install with: uv add pymupdf"
        )

    source = str(path.relative_to(docs_root))
    all_chunks: list[Chunk] = []
    global_idx = 0

    with fitz.open(str(path)) as doc:
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text("text").strip()
            if not page_text:
                continue
            page_chunks = chunk_text(
                page_text, source, chunk_size=chunk_size, overlap=overlap
            )
            for c in page_chunks:
                all_chunks.append(Chunk(
                    text=c.text,
                    source=source,
                    chunk_index=global_idx,
                    heading=f"Page {page_num}",
                ))
                global_idx += 1

    return all_chunks


def chunk_file(
    path: Path,
    docs_root: Path,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[Chunk]:
    """
    Read *path* and return its chunks. *source* is relative to *docs_root*.
    Dispatches to chunk_pdf for .pdf files, otherwise reads as text.
    """
    if path.suffix.lower() == ".pdf":
        return chunk_pdf(path, docs_root, chunk_size=chunk_size, overlap=overlap)
    text = path.read_text(encoding="utf-8", errors="replace")
    source = str(path.relative_to(docs_root))
    return chunk_text(text, source, chunk_size=chunk_size, overlap=overlap)
