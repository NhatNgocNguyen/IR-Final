"""
retriever.py — Retrieval + context assembly.

Responsibilities:
  1. Embed the question
  2. Run hybrid search on VectorDB
  3. Assemble retrieved chunks into a context string
     that fits within MAX_CONTEXT_CHARS
  4. Optionally de-duplicate highly overlapping chunks
"""

import logging
from typing import List, Tuple

import numpy as np

from config import MAX_CONTEXT_CHARS, TOP_K_FINAL, TOP_K_RETRIEVE
from embedder import embed_query
from vectordb import VectorDB, get_db

log = logging.getLogger(__name__)

# Similarity threshold above which two chunks are considered duplicates
_DEDUP_THRESHOLD = 0.92


def retrieve_chunks(
    question: str,
    db: VectorDB | None = None,
    top_k: int = TOP_K_FINAL,
) -> List[str]:
    """
    Return the top-k most relevant chunks for `question`.

    Steps:
      1. Embed question
      2. Hybrid search (BM25 + dense)
      3. De-duplicate near-identical chunks
      4. Return top-k
    """
    if db is None:
        db = get_db()

    if not db.is_ready:
        log.error("VectorDB not ready")
        return []

    q_emb = embed_query(question)

    # Fetch a larger pool, then prune
    pool_k  = min(TOP_K_RETRIEVE, db.n)
    results = db.search(question, q_emb, top_k=pool_k)   # [(chunk, score)]

    # De-duplicate: skip chunks that are very similar to already-kept ones
    kept:      List[str]       = []
    kept_vecs: List[np.ndarray] = []

    for chunk, _score in results:
        if len(kept) >= top_k:
            break
        c_emb = _chunk_embedding(chunk, db)
        if c_emb is not None and kept_vecs:
            sims = np.array([float(c_emb @ v) for v in kept_vecs])
            if sims.max() > _DEDUP_THRESHOLD:
                continue  # skip near-duplicate
        kept.append(chunk)
        if c_emb is not None:
            kept_vecs.append(c_emb)

    return kept


def _chunk_embedding(chunk: str, db: VectorDB) -> np.ndarray | None:
    """Look up pre-computed embedding for a chunk (O(n) scan, acceptable for small DBs)."""
    if db.embeddings is None:
        return None
    try:
        idx = db.chunks.index(chunk)
        return db.embeddings[idx]
    except ValueError:
        return None


def build_context(chunks: List[str], max_chars: int = MAX_CONTEXT_CHARS) -> str:
    """
    Assemble chunks into a numbered context string within `max_chars`.

    Chunks that don't fit are truncated with "…" to maximise information
    density rather than dropping entirely.
    """
    parts: List[str] = []
    used = 0

    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] "
        full   = header + chunk + "\n"

        if used + len(full) <= max_chars:
            parts.append(full)
            used += len(full)
        else:
            remaining = max_chars - used - len(header) - 5
            if remaining > 60:
                parts.append(header + chunk[:remaining] + "…\n")
                used = max_chars
            break

    return "".join(parts).strip()
