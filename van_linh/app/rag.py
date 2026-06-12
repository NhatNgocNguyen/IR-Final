from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings


LOGGER = logging.getLogger(__name__)
LEGAL_SECTION_PATTERN = re.compile(
    r"(?=^(?:PHAN|Phan|CHUONG|Chuong|MUC|Muc|DIEU|Dieu|Phần|Chương|Mục|Điều)\b)",
    re.MULTILINE,
)
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
LEGAL_REFERENCE_PATTERN = re.compile(
    r"\b(?:dieu|dieu|khoan|muc|chuong|phan|điều|khoản|mục|chương|phần)\s+\d+\b",
    re.IGNORECASE,
)
STOPWORDS = {
    "a",
    "an",
    "and",
    "b",
    "c",
    "d",
    "cua",
    "cho",
    "co",
    "duoc",
    "hay",
    "khi",
    "khong",
    "la",
    "mot",
    "neu",
    "nhung",
    "theo",
    "thi",
    "trong",
    "tu",
    "ve",
    "và",
    "của",
    "cho",
    "có",
    "được",
    "hay",
    "khi",
    "không",
    "là",
    "một",
    "nếu",
    "những",
    "theo",
    "thì",
    "trong",
    "từ",
    "về",
}


def _normalize_text(text: str) -> str:
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _split_long_segment(segment: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size - chunk_overlap)
    while start < len(segment):
        candidate_end = min(len(segment), start + chunk_size)
        end = candidate_end
        if candidate_end < len(segment):
            newline_break = segment.rfind("\n", start, candidate_end)
            space_break = segment.rfind(" ", start, candidate_end)
            best_break = max(newline_break, space_break)
            if best_break > start + (chunk_size // 2):
                end = best_break
        chunk = segment[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(segment):
            break
        start += stride
    return chunks


def _segment_legal_text(normalized_text: str) -> list[str]:
    segments = [
        segment.strip()
        for segment in LEGAL_SECTION_PATTERN.split(normalized_text)
        if segment.strip()
    ]
    return segments or [normalized_text]


def _chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    segments = _segment_legal_text(normalized)
    if len(segments) == 1 and len(segments[0]) <= chunk_size:
        return segments

    chunks: list[str] = []
    current_segments: list[str] = []
    current_length = 0

    for segment in segments:
        if len(segment) > chunk_size:
            if current_segments:
                chunks.append("\n".join(current_segments))
                current_segments = []
                current_length = 0
            chunks.extend(_split_long_segment(segment, chunk_size, chunk_overlap))
            continue

        projected_length = current_length + len(segment) + (1 if current_segments else 0)
        if current_segments and projected_length > chunk_size:
            chunks.append("\n".join(current_segments))
            current_segments = [segment]
            current_length = len(segment)
            continue

        current_segments.append(segment)
        current_length = projected_length

    if current_segments:
        chunks.append("\n".join(current_segments))

    return chunks


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]


def _tokenize(text: str) -> set[str]:
    return {
        token
        for token in TOKEN_PATTERN.findall(text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def _legal_reference_bonus(question: str, chunk: str) -> float:
    question_refs = set(LEGAL_REFERENCE_PATTERN.findall(question))
    if not question_refs:
        return 0.0

    chunk_text = chunk.lower()
    matches = sum(1 for ref in question_refs if ref.lower() in chunk_text)
    return 0.12 * matches


def _lexical_score(question: str, chunk: str) -> float:
    question_tokens = _tokenize(question)
    if not question_tokens:
        return 0.0

    chunk_tokens = _tokenize(chunk)
    if not chunk_tokens:
        return 0.0

    overlap = len(question_tokens & chunk_tokens)
    return overlap / len(question_tokens)


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(
            str(settings.embedding_model_path),
            local_files_only=True,
        )
        self._lock = Lock()
        self._doc_id: str | None = None
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None
        LOGGER.info("Embedding model loaded from %s", settings.embedding_model_path)

    def ingest(self, text: str, doc_id: str | None) -> tuple[str | None, int]:
        chunks = _chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
        )
        if not chunks:
            raise ValueError("Document did not produce any chunks")

        LOGGER.info("Encoding %s chunks for doc_id=%s", len(chunks), doc_id)
        embeddings = self._model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)

        with self._lock:
            self._doc_id = doc_id
            self._chunks = chunks
            self._embeddings = embeddings

        return self._doc_id, len(chunks)

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        with self._lock:
            embeddings = self._embeddings
            chunks = list(self._chunks)

        if embeddings is None or not chunks:
            raise LookupError("No document has been uploaded yet")

        query_vector = self._model.encode(
            [question],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        dense_scores = embeddings @ query_vector
        candidate_count = min(len(chunks), max(top_k * 3, top_k + 5))
        candidate_indices = np.argsort(dense_scores)[::-1][:candidate_count]

        reranked = []
        for index in candidate_indices:
            chunk = chunks[index]
            lexical = _lexical_score(question, chunk)
            legal_bonus = _legal_reference_bonus(question, chunk)
            final_score = (0.78 * float(dense_scores[index])) + (0.22 * lexical) + legal_bonus
            reranked.append((final_score, index))

        reranked.sort(key=lambda item: item[0], reverse=True)
        top_indices = [index for _, index in reranked[:top_k]]
        selected_chunks = [chunks[index] for index in top_indices]
        selected_scores = [score for score, index in reranked[:top_k]]
        return RetrievalResult(chunks=selected_chunks, scores=selected_scores)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._embeddings is not None and bool(self._chunks)
