"""
src/mcp_server/rag/retriever.py
────────────────────────────────
Retrieves the most similar chunks from the ChromaDB collection
given a natural-language query.
"""

from __future__ import annotations

import os
from typing import Any

import chromadb
from chromadb.config import Settings
from openai import OpenAI

from .indexer import (
    COLLECTION_NAME,
    EMBED_MODEL,
    _db_dir,
    _get_client,
    _get_collection,
)

# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    n_results: int = 5,
    source_filter: str | None = None,
) -> dict[str, Any]:
    """
    Return the *n_results* most relevant chunks for *query*.

    Args:
        query: Natural-language question or search string.
        n_results: Maximum number of chunks to return.
        source_filter: If set, restrict results to this source filename
                       (e.g. "parameter_reference.md").

    Returns:
        dict with keys:
            status  : "ok" | "error"
            results : list of {text, source, heading, score}
            message : human-readable summary
    """
    api_key = os.getenv("OPENAI_API_KEY", "")
    if not api_key:
        return {"status": "error", "message": "OPENAI_API_KEY not set."}

    try:
        client = _get_client()
        collection = _get_collection(client)

        count = collection.count()
        if count == 0:
            return {
                "status": "error",
                "message": (
                    "The document index is empty. "
                    "Call index_documents first to build the index."
                ),
                "results": [],
            }

        # Embed the query
        openai_client = OpenAI(api_key=api_key)
        response = openai_client.embeddings.create(model=EMBED_MODEL, input=[query])
        query_embedding = response.data[0].embedding

        where = {"source": source_filter} if source_filter else None

        raw = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(n_results, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        results = []
        docs = raw["documents"][0] if raw["documents"] else []
        metas = raw["metadatas"][0] if raw["metadatas"] else []
        dists = raw["distances"][0] if raw["distances"] else []

        for doc, meta, dist in zip(docs, metas, dists):
            # ChromaDB cosine distance: 0 = identical, 2 = opposite.
            # Convert to similarity score in [0, 1].
            score = round(1.0 - dist / 2.0, 4)
            results.append({
                "text": doc,
                "source": meta.get("source", ""),
                "heading": meta.get("heading", ""),
                "score": score,
            })

        return {
            "status": "ok",
            "results": results,
            "message": f"Found {len(results)} relevant chunk(s) for query.",
        }

    except Exception as exc:
        return {"status": "error", "message": str(exc), "results": []}
