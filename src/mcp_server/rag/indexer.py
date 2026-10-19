"""
src/mcp_server/rag/indexer.py
──────────────────────────────
Reads docs/, chunks documents, generates OpenAI embeddings,
and persists them in a ChromaDB collection stored at data/db/.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .chunker import Chunk, chunk_file

# ── Constants ─────────────────────────────────────────────────────────────────

COLLECTION_NAME = "simuscan_docs"
EMBED_MODEL = "text-embedding-3-small"
SUPPORTED_EXTENSIONS = {".md", ".txt", ".rst", ".pdf"}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent.parent


def _docs_dir() -> Path:
    return _repo_root() / "docs"


def _file_hash(path: Path) -> str:
    """SHA-1 of the file's raw bytes — used to detect content changes."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def _db_dir() -> Path:
    db = _repo_root() / "data" / "db"
    db.mkdir(parents=True, exist_ok=True)
    return db


def _get_client() -> chromadb.ClientAPI:
    return chromadb.PersistentClient(
        path=str(_db_dir()),
        settings=Settings(anonymized_telemetry=False),
    )


def _get_collection(client: chromadb.ClientAPI) -> chromadb.Collection:
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def _embed(texts: list[str], api_key: str) -> list[list[float]]:
    openai_client = OpenAI(api_key=api_key)
    response = openai_client.embeddings.create(model=EMBED_MODEL, input=texts)
    return [item.embedding for item in response.data]


# ── Public API ────────────────────────────────────────────────────────────────

def index_docs_folder(
    chunk_size: int = 512,
    overlap: int = 64,
    force_reindex: bool = False,
) -> dict[str, Any]:
    """
    Index all supported documents in docs/.
    Returns a summary dict with counts.
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "OPENAI_API_KEY not set."}

    docs_dir = _docs_dir()
    if not docs_dir.exists():
        return {"status": "error", "message": f"docs/ directory not found at {docs_dir}"}

    files = [
        p for p in docs_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    if not files:
        return {"status": "ok", "message": "No documents found in docs/.", "indexed": 0}

    client = _get_client()
    collection = _get_collection(client)

    if force_reindex:
        client.delete_collection(COLLECTION_NAME)
        collection = _get_collection(client)

    # Map each already-indexed source to its stored content hash (from any of its
    # chunks). Used to skip unchanged files and re-index changed ones.
    stored_meta = collection.get(include=["metadatas"])["metadatas"] or []
    source_hashes: dict[str, str | None] = {}
    for m in stored_meta:
        src = m.get("source")
        if src is not None and src not in source_hashes:
            source_hashes[src] = m.get("content_hash")

    total_chunks = 0
    skipped_files = 0     # unchanged, already indexed
    new_files = 0         # not previously indexed
    changed_files = 0     # content changed → re-indexed

    for file_path in files:
        source = str(file_path.relative_to(docs_dir))
        file_hash = _file_hash(file_path)
        already = source in source_hashes
        unchanged = already and source_hashes.get(source) == file_hash

        if unchanged and not force_reindex:
            skipped_files += 1
            continue

        # Remove stale chunks for this source before re-adding
        existing = collection.get(where={"source": source})
        if existing["ids"]:
            collection.delete(ids=existing["ids"])

        chunks: list[Chunk] = chunk_file(
            file_path, docs_dir, chunk_size=chunk_size, overlap=overlap
        )
        if not chunks:
            continue

        texts = [c.text for c in chunks]
        embeddings = _embed(texts, api_key)

        ids = [f"{source}::{c.chunk_index}" for c in chunks]
        metadatas = [
            {"source": c.source, "chunk_index": c.chunk_index,
             "heading": c.heading, "content_hash": file_hash}
            for c in chunks
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas,
        )
        total_chunks += len(chunks)
        if already:
            changed_files += 1
        else:
            new_files += 1

    return {
        "status": "ok",
        "message": (
            f"Indexed {new_files} new + {changed_files} changed file(s) -> "
            f"{total_chunks} chunk(s). Skipped {skipped_files} unchanged file(s)."
        ),
        "new_files": new_files,
        "changed_files": changed_files,
        "skipped_files": skipped_files,
        "total_chunks": total_chunks,
    }


def index_single_file(
    file_path: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> dict[str, Any]:
    """Index or re-index a single file (absolute or relative to docs/)."""
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "OPENAI_API_KEY not set."}

    docs_dir = _docs_dir()
    path = Path(file_path)
    if not path.is_absolute():
        path = docs_dir / file_path
    if not path.exists():
        return {"status": "error", "message": f"File not found: {path}"}

    client = _get_client()
    collection = _get_collection(client)

    source = str(path.relative_to(docs_dir))
    existing = collection.get(where={"source": source})
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    chunks = chunk_file(path, docs_dir, chunk_size=chunk_size, overlap=overlap)
    if not chunks:
        return {"status": "ok", "message": "File produced no chunks.", "chunks": 0}

    file_hash = _file_hash(path)
    texts = [c.text for c in chunks]
    embeddings = _embed(texts, api_key)
    ids = [f"{source}::{c.chunk_index}" for c in chunks]
    metadatas = [
        {"source": c.source, "chunk_index": c.chunk_index,
         "heading": c.heading, "content_hash": file_hash}
        for c in chunks
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)

    return {
        "status": "ok",
        "message": f"Indexed '{source}' -> {len(chunks)} chunk(s).",
        "chunks": len(chunks),
    }


def list_indexed_documents() -> dict[str, Any]:
    """Return the list of source files currently in the vector DB."""
    try:
        client = _get_client()
        collection = _get_collection(client)
        metadatas = collection.get(include=["metadatas"])["metadatas"] or []
        sources = sorted({m["source"] for m in metadatas})
        return {
            "status": "ok",
            "documents": sources,
            "total_chunks": len(metadatas),
        }
    except Exception as exc:
        return {"status": "error", "message": str(exc)}
