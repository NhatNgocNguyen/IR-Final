from __future__ import annotations

import json
import logging
import math
import re
import time
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from threading import Lock

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import Settings
from app.text_utils import strip_options


LOGGER = logging.getLogger(__name__)
RETRIEVAL_VERSION = 4
TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)
REFERENCE_PATTERN = re.compile(
    r"\b(?:điều|dieu|khoản|khoan|mục|muc|"
    r"chương|chuong|phần|phan|"
    r"section|chapter|part|article|clause)\s+\d+(?:\.\d+)*\b",
    re.IGNORECASE,
)
NUMBERED_HEADING_PATTERN = re.compile(
    r"^(?:\d+(?:\.\d+){0,5}|[IVXLCDM]+)[\.\)]?\s+\S+",
    re.IGNORECASE,
)
LEGAL_HEADING_PATTERN = re.compile(
    r"^(?:PHẦN|PHAN|CHƯƠNG|CHUONG|MỤC|MUC|ĐIỀU|DIEU)\s+\S+",
    re.IGNORECASE,
)
BULLET_PATTERN = re.compile(r"^(?:[-+*•]|\(?[a-zA-Z]\)|[a-zA-Z][\.\)])\s+\S+")
SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?;])\s+|\n+")
STOPWORDS = {
    "a",
    "an",
    "and",
    "b",
    "c",
    "d",
    "các",
    "cho",
    "có",
    "cua",
    "của",
    "đã",
    "đó",
    "được",
    "duoc",
    "hay",
    "hoặc",
    "khi",
    "không",
    "la",
    "là",
    "mot",
    "một",
    "này",
    "neu",
    "nếu",
    "nhung",
    "những",
    "sẽ",
    "theo",
    "thi",
    "thì",
    "trong",
    "tu",
    "từ",
    "ve",
    "về",
    "và",
    "với",
}


@dataclass
class RetrievalResult:
    chunks: list[str]
    scores: list[float]


@dataclass
class RagIndexState:
    doc_id: str | None
    chunk_count: int
    ready: bool
    index_persisted: bool


@dataclass
class LexicalIndex:
    token_counts: list[Counter[str]]
    idf: dict[str, float]
    average_length: float


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize(
        "NFC",
        text.replace("\r\n", "\n").replace("\r", "\n"),
    )
    return "\n".join(line.strip() for line in normalized.splitlines() if line.strip())


def _is_uppercase_heading(line: str) -> bool:
    if len(line) < 8 or len(line) > 180:
        return False
    letters = [character for character in line if character.isalpha()]
    if len(letters) < 6:
        return False
    uppercase_letters = [
        character for character in letters if character.upper() == character
    ]
    return len(uppercase_letters) / len(letters) >= 0.75


def _is_heading(line: str) -> bool:
    cleaned = line.strip()
    if not cleaned:
        return False
    return (
        bool(LEGAL_HEADING_PATTERN.match(cleaned))
        or bool(NUMBERED_HEADING_PATTERN.match(cleaned))
        or _is_uppercase_heading(cleaned)
    )


def _split_long_segment(segment: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    chunks: list[str] = []
    start = 0
    stride = max(1, chunk_size - chunk_overlap)
    while start < len(segment):
        candidate_end = min(len(segment), start + chunk_size)
        end = candidate_end
        if candidate_end < len(segment):
            newline_break = segment.rfind("\n", start, candidate_end)
            sentence_break = max(
                segment.rfind(". ", start, candidate_end),
                segment.rfind("? ", start, candidate_end),
                segment.rfind("! ", start, candidate_end),
                segment.rfind("; ", start, candidate_end),
            )
            space_break = segment.rfind(" ", start, candidate_end)
            best_break = max(newline_break, sentence_break, space_break)
            if best_break > start + (chunk_size // 2):
                end = best_break + 1
        chunk = segment[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(segment):
            break
        start += stride
    return chunks


def _overlap_tail(text: str, overlap: int) -> str:
    if overlap <= 0 or not text:
        return ""
    if len(text) <= overlap:
        return text.strip()

    segment = text[-overlap:]
    for separator in ("\n", ". ", "? ", "! ", "; ", " "):
        position = segment.find(separator)
        if position >= 0:
            tail = segment[position + len(separator) :].strip()
            if tail:
                return tail
    return segment.strip()


def _segment_text(normalized_text: str) -> list[str]:
    lines = [line.strip() for line in normalized_text.splitlines() if line.strip()]
    if not lines:
        return []

    segments: list[str] = []
    current_lines: list[str] = []
    for line in lines:
        starts_new_section = bool(current_lines) and _is_heading(line)
        starts_new_bullet_block = (
            bool(current_lines)
            and len("\n".join(current_lines)) > 500
            and bool(BULLET_PATTERN.match(line))
        )
        if starts_new_section or starts_new_bullet_block:
            segments.append("\n".join(current_lines).strip())
            current_lines = [line]
            continue
        current_lines.append(line)

    if current_lines:
        segments.append("\n".join(current_lines).strip())

    return segments or [normalized_text]


def _token_len(tokenizer, text: str) -> int:
    stripped = text.strip()
    if not stripped:
        return 0
    return len(tokenizer(stripped, add_special_tokens=False)["input_ids"])


def _split_into_sentences(text: str) -> list[str]:
    parts = SENTENCE_SPLIT_PATTERN.split(text.strip())
    return [part.strip() for part in parts if part.strip()]


def _join_text_parts(parts: list[str]) -> str:
    if any("\n" in part for part in parts):
        return "\n".join(parts)
    return " ".join(parts)


def _overlap_tail_tokens(parts: list[str], tokenizer, overlap_tokens: int) -> list[str]:
    if overlap_tokens <= 0 or not parts:
        return []

    collected: list[str] = []
    for part in reversed(parts):
        collected.insert(0, part)
        if _token_len(tokenizer, _join_text_parts(collected)) > overlap_tokens:
            collected.pop(0)
            break
    return collected or [parts[-1]]


def _split_by_words(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    words = text.split()
    if not words:
        return []

    chunks: list[str] = []
    current_words: list[str] = []

    def flush_current() -> None:
        nonlocal current_words
        if current_words:
            chunks.append(" ".join(current_words))
            current_words = []

    for word in words:
        candidate = " ".join(current_words + [word]) if current_words else word
        if current_words and _token_len(tokenizer, candidate) > max_tokens:
            flush_current()
            if _token_len(tokenizer, word) > max_tokens:
                chunks.append(word)
                continue
            if chunks and overlap_tokens > 0:
                previous_words = chunks[-1].split()
                overlap_words: list[str] = []
                for previous_word in reversed(previous_words):
                    overlap_candidate = " ".join(reversed(overlap_words) + [previous_word])
                    if overlap_words and _token_len(tokenizer, overlap_candidate) > overlap_tokens:
                        break
                    overlap_words.insert(0, previous_word)
                current_words = overlap_words
            current_words.append(word)
            continue
        current_words.append(word)

    flush_current()
    return chunks


def _split_by_tokens(
    text: str,
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if _token_len(tokenizer, stripped) <= max_tokens:
        return [stripped]

    sentences = _split_into_sentences(stripped)
    if len(sentences) == 1:
        return _split_by_words(stripped, tokenizer, max_tokens, overlap_tokens)

    chunks: list[str] = []
    current_parts: list[str] = []

    def flush_with_overlap() -> None:
        nonlocal current_parts
        if not current_parts:
            return
        chunks.append(_join_text_parts(current_parts))
        if overlap_tokens > 0:
            current_parts = _overlap_tail_tokens(current_parts, tokenizer, overlap_tokens)
        else:
            current_parts = []

    for sentence in sentences:
        if _token_len(tokenizer, sentence) > max_tokens:
            if current_parts:
                chunks.append(_join_text_parts(current_parts))
                current_parts = []
            chunks.extend(_split_by_words(sentence, tokenizer, max_tokens, overlap_tokens))
            continue

        projected_parts = current_parts + [sentence]
        projected_text = _join_text_parts(projected_parts)
        if current_parts and _token_len(tokenizer, projected_text) > max_tokens:
            flush_with_overlap()
            current_parts.append(sentence)
            continue

        current_parts.append(sentence)

    if current_parts:
        chunks.append(_join_text_parts(current_parts))

    return chunks


def _apply_token_guard(
    chunks: list[str],
    tokenizer,
    max_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    guarded: list[str] = []
    for chunk in chunks:
        stripped = chunk.strip()
        if not stripped:
            continue
        if _token_len(tokenizer, stripped) <= max_tokens:
            guarded.append(stripped)
        else:
            guarded.extend(_split_by_tokens(stripped, tokenizer, max_tokens, overlap_tokens))
    return guarded


def _chunk_text(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    *,
    tokenizer,
    max_chunk_tokens: int,
) -> list[str]:
    normalized = _normalize_text(text)
    if not normalized:
        return []

    overlap_tokens = max(16, max_chunk_tokens // 8)
    segments = _segment_text(normalized)
    if len(segments) == 1 and len(segments[0]) <= chunk_size:
        chunks = segments
    else:
        chunks = []
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

            projected_length = current_length + len(segment) + (
                1 if current_segments else 0
            )
            if current_segments and projected_length > chunk_size:
                flushed = "\n".join(current_segments)
                chunks.append(flushed)
                tail = _overlap_tail(flushed, chunk_overlap)
                current_segments = [tail, segment] if tail else [segment]
                current_length = sum(len(part) for part in current_segments) + max(
                    0, len(current_segments) - 1
                )
                continue

            current_segments.append(segment)
            current_length = projected_length

        if current_segments:
            chunks.append("\n".join(current_segments))

    return _apply_token_guard(chunks, tokenizer, max_chunk_tokens, overlap_tokens)


def _tokenize_to_list(text: str) -> list[str]:
    normalized = unicodedata.normalize("NFC", text.lower())
    return [
        token
        for token in TOKEN_PATTERN.findall(normalized)
        if len(token) > 1 and token not in STOPWORDS
    ]


def _tokenize(text: str) -> set[str]:
    return set(_tokenize_to_list(text))


def _build_lexical_index(chunks: list[str]) -> LexicalIndex:
    token_counts = [Counter(_tokenize_to_list(chunk)) for chunk in chunks]
    if not token_counts:
        return LexicalIndex(token_counts=[], idf={}, average_length=0.0)

    document_frequency: Counter[str] = Counter()
    for counts in token_counts:
        document_frequency.update(counts.keys())

    chunk_count = len(token_counts)
    idf = {
        token: math.log(1 + ((chunk_count - frequency + 0.5) / (frequency + 0.5)))
        for token, frequency in document_frequency.items()
    }
    average_length = sum(sum(counts.values()) for counts in token_counts) / chunk_count
    return LexicalIndex(
        token_counts=token_counts,
        idf=idf,
        average_length=average_length,
    )


def _bm25_score(
    query_tokens: set[str],
    chunk_counts: Counter[str],
    lexical_index: LexicalIndex,
) -> float:
    if not query_tokens or not chunk_counts or lexical_index.average_length <= 0:
        return 0.0

    k1 = 1.5
    b = 0.75
    chunk_length = sum(chunk_counts.values())
    score = 0.0
    for token in query_tokens:
        frequency = chunk_counts.get(token, 0)
        if frequency == 0:
            continue
        denominator = frequency + k1 * (
            1 - b + b * (chunk_length / lexical_index.average_length)
        )
        score += lexical_index.idf.get(token, 0.0) * (
            (frequency * (k1 + 1)) / denominator
        )
    return score


def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
    if not scores:
        return {}
    values = list(scores.values())
    minimum = min(values)
    maximum = max(values)
    if maximum <= minimum:
        return {index: 1.0 for index in scores}
    return {
        index: (score - minimum) / (maximum - minimum)
        for index, score in scores.items()
    }


def _reference_bonus(question: str, chunk: str) -> float:
    question_refs = set(REFERENCE_PATTERN.findall(question))
    if not question_refs:
        return 0.0

    matches = sum(
        1
        for ref in question_refs
        if re.search(r"\b" + re.escape(ref) + r"\b", chunk, re.IGNORECASE)
    )
    return 0.12 * matches


def _exact_phrase_bonus(question_tokens: set[str], chunk: str) -> float:
    if not question_tokens:
        return 0.0
    chunk_text = chunk.lower()
    important_tokens = [token for token in question_tokens if len(token) >= 5]
    matches = sum(1 for token in important_tokens if token in chunk_text)
    return min(0.15, 0.015 * matches)


def _select_mmr(
    reranked: list[tuple[float, int]],
    embeddings: np.ndarray,
    top_k: int,
    mmr_lambda: float,
    dedup_threshold: float,
) -> tuple[list[int], dict[int, float]]:
    if not reranked:
        return [], {}

    relevance = _normalize_scores({index: score for score, index in reranked})
    remaining = [index for _, index in reranked]
    selected: list[int] = []
    selected_scores: dict[int, float] = {}

    while len(selected) < top_k and remaining:
        if not selected:
            best_index = max(remaining, key=lambda index: relevance[index])
            best_mmr = relevance[best_index]
        else:
            best_index = None
            best_mmr = float("-inf")
            for index in remaining:
                max_similarity = max(
                    float(embeddings[index] @ embeddings[chosen])
                    for chosen in selected
                )
                mmr_score = mmr_lambda * relevance[index] - (1 - mmr_lambda) * max_similarity
                if mmr_score > best_mmr:
                    best_mmr = mmr_score
                    best_index = index

        if best_index is None:
            break

        remaining.remove(best_index)
        if selected:
            max_similarity = max(
                float(embeddings[best_index] @ embeddings[chosen])
                for chosen in selected
            )
            if max_similarity > dedup_threshold:
                continue

        selected.append(best_index)
        selected_scores[best_index] = relevance[best_index]

    return selected, selected_scores


def _expand_neighbors(
    core_indices: list[int],
    core_scores: dict[int, float],
    *,
    total_chunks: int,
    top_k: int,
    neighbor_radius: int,
) -> tuple[list[int], dict[int, float]]:
    if neighbor_radius <= 0 or not core_indices:
        return core_indices, core_scores

    expanded_indices: set[int] = set()
    for index in core_indices:
        for offset in range(-neighbor_radius, neighbor_radius + 1):
            neighbor_index = index + offset
            if 0 <= neighbor_index < total_chunks:
                expanded_indices.add(neighbor_index)

    max_chunks = top_k + 2 * neighbor_radius * top_k
    ordered_indices = sorted(expanded_indices)
    if len(ordered_indices) > max_chunks:
        ordered_indices = ordered_indices[:max_chunks]

    expanded_scores: dict[int, float] = {}
    for index in ordered_indices:
        if index in core_scores:
            expanded_scores[index] = core_scores[index]
            continue
        nearest_core = min(core_indices, key=lambda core: abs(core - index))
        expanded_scores[index] = core_scores[nearest_core]

    return ordered_indices, expanded_scores


class RagService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = SentenceTransformer(
            settings.embedding_model_source,
            local_files_only=settings.embedding_model_local_only,
        )
        self._index_dir = settings.index_storage_dir
        self._metadata_path = self._index_dir / "metadata.json"
        self._embeddings_path = self._index_dir / "embeddings.npy"
        self._lock = Lock()
        self._doc_id: str | None = None
        self._chunks: list[str] = []
        self._embeddings: np.ndarray | None = None
        self._lexical_index = LexicalIndex(token_counts=[], idf={}, average_length=0.0)
        self._index_persisted = False
        LOGGER.info(
            "Embedding model loaded from %s (local_only=%s)",
            settings.embedding_model_source,
            settings.embedding_model_local_only,
        )
        self._load_persisted_index()

    def ingest(self, text: str, doc_id: str | None) -> tuple[str | None, int]:
        chunks = _chunk_text(
            text,
            chunk_size=self._settings.chunk_size,
            chunk_overlap=self._settings.chunk_overlap,
            tokenizer=self._model.tokenizer,
            max_chunk_tokens=self._settings.max_chunk_tokens,
        )
        if not chunks:
            raise ValueError("Document did not produce any chunks")

        start_time = time.perf_counter()
        LOGGER.info("Encoding %s chunks for doc_id=%s", len(chunks), doc_id)
        embeddings = self._model.encode(
            chunks,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)
        lexical_index = _build_lexical_index(chunks)

        self._persist_index(doc_id=doc_id, text=text, chunks=chunks, embeddings=embeddings)

        with self._lock:
            self._doc_id = doc_id
            self._chunks = chunks
            self._embeddings = embeddings
            self._lexical_index = lexical_index
            self._index_persisted = True

        LOGGER.info(
            "Indexed %s chunks for doc_id=%s in %.3fs",
            len(chunks),
            doc_id,
            time.perf_counter() - start_time,
        )
        return self._doc_id, len(chunks)

    def retrieve(self, question: str, top_k: int) -> RetrievalResult:
        started_at = time.perf_counter()
        with self._lock:
            embeddings = self._embeddings
            chunks = list(self._chunks)
            lexical_index = self._lexical_index

        if embeddings is None or not chunks:
            raise LookupError("No document has been uploaded yet")

        clean_query = strip_options(question) or question
        query_vector = self._model.encode(
            [clean_query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        dense_scores = embeddings @ query_vector
        question_tokens = _tokenize(clean_query)

        candidate_count = min(len(chunks), max(top_k * 4, top_k + 8))
        candidate_indices = np.argsort(dense_scores)[::-1][:candidate_count]
        candidate_indices_list = [int(index) for index in candidate_indices]

        dense_candidate_scores = {
            index: float(dense_scores[index]) for index in candidate_indices_list
        }
        bm25_candidate_scores = {
            index: _bm25_score(
                question_tokens,
                lexical_index.token_counts[index],
                lexical_index,
            )
            for index in candidate_indices_list
        }
        dense_norm = _normalize_scores(dense_candidate_scores)
        bm25_norm = _normalize_scores(bm25_candidate_scores)

        reranked: list[tuple[float, int]] = []
        for index in candidate_indices_list:
            chunk = chunks[index]
            final_score = (
                0.70 * dense_norm[index]
                + 0.30 * bm25_norm[index]
                + _reference_bonus(question, chunk)
                + _exact_phrase_bonus(question_tokens, chunk)
            )
            reranked.append((final_score, index))

        reranked.sort(key=lambda item: item[0], reverse=True)
        core_indices, core_scores = _select_mmr(
            reranked,
            embeddings,
            top_k,
            self._settings.mmr_lambda,
            self._settings.dedup_threshold,
        )
        selected_indices, selected_score_map = _expand_neighbors(
            core_indices,
            core_scores,
            total_chunks=len(chunks),
            top_k=top_k,
            neighbor_radius=self._settings.neighbor_radius,
        )
        selected_chunks = [chunks[index] for index in selected_indices]
        selected_scores = [selected_score_map[index] for index in selected_indices]

        LOGGER.info(
            "Retrieved %s chunks (%s core) from %s candidates in %.3fs; top_scores=%s",
            len(selected_chunks),
            len(core_indices),
            candidate_count,
            time.perf_counter() - started_at,
            [round(score, 4) for score, _ in reranked[:5]],
        )
        return RetrievalResult(chunks=selected_chunks, scores=selected_scores)

    @property
    def ready(self) -> bool:
        with self._lock:
            return self._embeddings is not None and bool(self._chunks)

    @property
    def index_state(self) -> RagIndexState:
        with self._lock:
            return RagIndexState(
                doc_id=self._doc_id,
                chunk_count=len(self._chunks),
                ready=self._embeddings is not None and bool(self._chunks),
                index_persisted=self._index_persisted,
            )

    def _persist_index(
        self,
        *,
        doc_id: str | None,
        text: str,
        chunks: list[str],
        embeddings: np.ndarray,
    ) -> None:
        self._index_dir.mkdir(parents=True, exist_ok=True)
        metadata = {
            "doc_id": doc_id,
            "chunks": chunks,
            "embedding_model_path": self._settings.embedding_model_source,
            "chunk_size": self._settings.chunk_size,
            "chunk_overlap": self._settings.chunk_overlap,
            "max_chunk_tokens": self._settings.max_chunk_tokens,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "text_hash": sha256(text.encode("utf-8")).hexdigest(),
            "retrieval_version": RETRIEVAL_VERSION,
        }

        metadata_tmp_path = self._metadata_path.with_name("metadata.tmp.json")
        embeddings_tmp_path = self._embeddings_path.with_name("embeddings.tmp.npy")

        with metadata_tmp_path.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
        np.save(embeddings_tmp_path, embeddings)

        metadata_tmp_path.replace(self._metadata_path)
        embeddings_tmp_path.replace(self._embeddings_path)
        LOGGER.info("Persisted index to %s", self._index_dir)

    def _load_persisted_index(self) -> None:
        if not self._metadata_path.exists() or not self._embeddings_path.exists():
            LOGGER.info("No persisted index found in %s", self._index_dir)
            return

        try:
            metadata = self._load_metadata()
            chunks = metadata["chunks"]
            embeddings = np.load(self._embeddings_path).astype(np.float32)
            if embeddings.ndim != 2:
                raise ValueError("Persisted embeddings must be a 2D array")
            if len(chunks) != int(embeddings.shape[0]):
                raise ValueError("Persisted chunks and embeddings row count do not match")
            lexical_index = _build_lexical_index(list(chunks))
        except Exception:
            LOGGER.exception("Failed to load persisted index from %s", self._index_dir)
            return

        with self._lock:
            self._doc_id = metadata["doc_id"]
            self._chunks = list(chunks)
            self._embeddings = embeddings
            self._lexical_index = lexical_index
            self._index_persisted = True

        LOGGER.info(
            "Loaded persisted index from %s with %s chunks",
            self._index_dir,
            len(chunks),
        )

    def _load_metadata(self) -> dict[str, object]:
        with self._metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)

        required_keys = {
            "doc_id",
            "chunks",
            "embedding_model_path",
            "chunk_size",
            "chunk_overlap",
            "created_at",
            "text_hash",
        }
        missing_keys = required_keys - metadata.keys()
        if missing_keys:
            raise ValueError(f"Persisted metadata missing keys: {sorted(missing_keys)}")

        if metadata["embedding_model_path"] != self._settings.embedding_model_source:
            raise ValueError("Persisted index embedding model does not match current config")
        if int(metadata["chunk_size"]) != self._settings.chunk_size:
            raise ValueError("Persisted index chunk_size does not match current config")
        if int(metadata["chunk_overlap"]) != self._settings.chunk_overlap:
            raise ValueError("Persisted index chunk_overlap does not match current config")
        if int(metadata.get("max_chunk_tokens", 0)) != self._settings.max_chunk_tokens:
            raise ValueError("Persisted index max_chunk_tokens does not match current config")
        if not isinstance(metadata["chunks"], list) or not all(
            isinstance(chunk, str) for chunk in metadata["chunks"]
        ):
            raise ValueError("Persisted chunks must be a list of strings")
        if int(metadata.get("retrieval_version", 0)) != RETRIEVAL_VERSION:
            raise ValueError(
                f"Persisted retrieval_version does not match current version {RETRIEVAL_VERSION}"
            )

        return metadata
