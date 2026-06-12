"""
vectordb.py — Hybrid VectorDB: BM25 + dense (cosine) retrieval.

Design:
  • BM25     → great for exact Vietnamese keyword matching
  • Dense    → great for semantic / paraphrase matching
  • Fusion   → weighted score combination (Reciprocal Rank Fusion variant)
  • Persist  → pickle to DATA_DIR/vectordb.pkl so restarts are instant

Thread-safety: single-worker uvicorn is assumed (no lock needed).
"""

import logging
import os
import pickle
import re
from typing import List, Optional, Tuple

import numpy as np

from config import (
    BM25_WEIGHT,
    DATA_DIR,
    DENSE_WEIGHT,
    TOP_K_FINAL,
    TOP_K_RETRIEVE,
    VECTORDB_PATH,
)

log = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Vietnamese tokenizer (space-based + lowercase + remove punctuation)
# ------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[^\w\s]", re.UNICODE)

def _tokenize(text: str) -> List[str]:
    """Lowercase, strip punctuation, split on whitespace."""
    text = text.lower()
    text = _TOKEN_RE.sub(" ", text)
    return text.split()


# ------------------------------------------------------------------
# BM25 helper (inline to avoid import issues if rank_bm25 missing)
# ------------------------------------------------------------------

def _safe_bm25_scores(bm25, tokens: List[str], n: int) -> np.ndarray:
    try:
        scores = np.array(bm25.get_scores(tokens), dtype=np.float32)
    except Exception:
        scores = np.zeros(n, dtype=np.float32)
    return scores


# ------------------------------------------------------------------
# VectorDB
# ------------------------------------------------------------------

class VectorDB:
    """
    Holds chunks, their embeddings, and a BM25 index.
    Call `add_documents` to populate, `search` to query, `save`/`load`
    to persist across server restarts.
    """

    def __init__(self) -> None:
        self.chunks:      List[str]            = []
        self.embeddings:  Optional[np.ndarray] = None
        self.doc_id:      Optional[str]        = None
        self._bm25                             = None
        self._tokenized:  List[List[str]]      = []

    # ---------------------------------------------------------------- state

    @property
    def is_ready(self) -> bool:
        return len(self.chunks) > 0

    @property
    def n(self) -> int:
        return len(self.chunks)

    # ---------------------------------------------------------------- build

    def add_documents(
        self,
        chunks: List[str],
        embeddings: np.ndarray,
        doc_id: str = "doc",
    ) -> None:
        """Replace current index with new chunks + embeddings."""
        self.chunks     = chunks
        self.embeddings = embeddings.astype(np.float32)
        self.doc_id     = doc_id

        # BM25 index
        self._tokenized = [_tokenize(c) for c in chunks]
        try:
            from rank_bm25 import BM25Okapi
            self._bm25 = BM25Okapi(self._tokenized)
            log.info("BM25 index built over %d chunks", len(chunks))
        except ImportError:
            log.warning("rank_bm25 not installed — using dense-only retrieval")
            self._bm25 = None

    # ---------------------------------------------------------------- search

    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        top_k: int = TOP_K_FINAL,
    ) -> List[Tuple[str, float]]:
        """
        Hybrid search.  Returns list of (chunk_text, score) sorted by score desc.
        """
        if not self.is_ready:
            return []

        n     = self.n
        score = np.zeros(n, dtype=np.float32)

        # ── Dense (cosine via dot product on L2-normed vectors) ──────────
        if self.embeddings is not None:
            qv = query_embedding.astype(np.float32)
            dense = self.embeddings @ qv            # shape (n,)
            dense = _minmax(dense)
            score += DENSE_WEIGHT * dense

        # ── BM25 ─────────────────────────────────────────────────────────
        if self._bm25 is not None:
            tokens   = _tokenize(query)
            bm25_raw = _safe_bm25_scores(self._bm25, tokens, n)
            bm25_raw = _minmax(bm25_raw)
            score   += BM25_WEIGHT * bm25_raw

        # ── Top-k ────────────────────────────────────────────────────────
        k       = min(top_k, n)
        indices = np.argpartition(score, -k)[-k:]
        indices = indices[np.argsort(score[indices])[::-1]]

        return [(self.chunks[i], float(score[i])) for i in indices]

    # ---------------------------------------------------------------- persistence

    def save(self, path: str = VECTORDB_PATH) -> None:
        os.makedirs(DATA_DIR, exist_ok=True)
        payload = {
            "chunks":     self.chunks,
            "embeddings": self.embeddings,
            "doc_id":     self.doc_id,
            "tokenized":  self._tokenized,
        }
        with open(path, "wb") as fh:
            pickle.dump(payload, fh, protocol=pickle.HIGHEST_PROTOCOL)
        log.info("VectorDB saved → %s (%d chunks)", path, len(self.chunks))

    def load(self, path: str = VECTORDB_PATH) -> bool:
        """Load from disk.  Returns True on success."""
        if not os.path.exists(path):
            return False
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
            self.chunks     = data["chunks"]
            self.embeddings = data["embeddings"]
            self.doc_id     = data.get("doc_id")
            self._tokenized = data.get("tokenized", [])

            # Rebuild BM25
            if self._tokenized:
                try:
                    from rank_bm25 import BM25Okapi
                    self._bm25 = BM25Okapi(self._tokenized)
                except ImportError:
                    self._bm25 = None

            log.info("VectorDB loaded ← %s (%d chunks)", path, len(self.chunks))
            return True
        except Exception as exc:
            log.error("Failed to load VectorDB: %s", exc)
            return False


# ------------------------------------------------------------------
# Module-level singleton
# ------------------------------------------------------------------

_db = VectorDB()

def get_db() -> VectorDB:
    return _db


# ------------------------------------------------------------------
# Utility
# ------------------------------------------------------------------

def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    if hi > lo:
        return (arr - lo) / (hi - lo)
    return np.zeros_like(arr)
