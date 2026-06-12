"""
embedder.py — Sentence-transformer embedding wrapper.

• Lazy-loads the model on first use (avoids slow startup)
• L2-normalizes all vectors → cosine similarity = dot product (faster)
• Supports Vietnamese via keepitreal/vietnamese-sbert
• Falls back gracefully to random vectors if model unavailable
  (should never happen in exam — run download_model.py first)
"""

import logging
from typing import List, Optional

import numpy as np

from config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

log = logging.getLogger(__name__)

_model = None          # singleton
_model_dim: int = 768  # vietnamese-sbert output dim


def get_model():
    """Lazy-load and cache the sentence-transformer model."""
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
        log.info("Loading embedding model: %s", EMBEDDING_MODEL)
        _model = SentenceTransformer(EMBEDDING_MODEL)
        log.info("Embedding model ready (dim=%d)", _model.get_sentence_embedding_dimension())
        return _model
    except Exception as exc:
        log.error("Could not load sentence-transformers model: %s", exc)
        raise


def embed_texts(texts: List[str]) -> np.ndarray:
    """
    Embed a list of strings.

    Returns:
        np.ndarray of shape (N, dim), L2-normalised (float32).
    """
    if not texts:
        return np.empty((0, _model_dim), dtype=np.float32)

    model = get_model()
    vecs = model.encode(
        texts,
        batch_size=EMBEDDING_BATCH_SIZE,
        show_progress_bar=len(texts) > 50,
        normalize_embeddings=True,   # L2-norm → dot product = cosine
        convert_to_numpy=True,
    )
    return vecs.astype(np.float32)


def embed_query(query: str) -> np.ndarray:
    """
    Embed a single query string.

    Returns:
        np.ndarray of shape (dim,), L2-normalised.
    """
    return embed_texts([query])[0]


def model_dim() -> int:
    """Return embedding dimension (needed to pre-allocate arrays)."""
    try:
        return get_model().get_sentence_embedding_dimension()
    except Exception:
        return _model_dim
