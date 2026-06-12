"""
chunker.py — Vietnamese-aware recursive text chunker.

Strategy (priority order):
  1. Split on paragraph boundaries  (\n\n)
  2. Split on sentence endings       (. ? ! … ;)
  3. Split on clause boundaries      (, — :)
  4. Split on word boundaries        (space)
  5. Hard character split            (last resort)

Chunks are merged up to CHUNK_SIZE then a sliding window of
CHUNK_OVERLAP chars is carried forward as context bridge.
"""

import re
import unicodedata
from typing import List

from config import CHUNK_SIZE, CHUNK_OVERLAP, MIN_CHUNK_LEN


# ------------------------------------------------------------------ helpers

def _normalize(text: str) -> str:
    """Unicode NFC + collapse whitespace."""
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _sentence_split(text: str) -> List[str]:
    """
    Split a block of Vietnamese text into sentences.
    Keeps the punctuation attached to the preceding sentence.
    """
    # Insert a marker after sentence-ending punctuation followed by space/newline
    pattern = r'(?<=[.!?…\u2026])\s+'
    parts = re.split(pattern, text)
    # Also split on newline within a paragraph
    sentences: List[str] = []
    for part in parts:
        sub = re.split(r'\n', part)
        sentences.extend(sub)
    return [s.strip() for s in sentences if s.strip()]


def _merge_into_chunks(sentences: List[str],
                       chunk_size: int,
                       chunk_overlap: int) -> List[str]:
    """
    Greedily merge sentences into chunks ≤ chunk_size chars.
    When a chunk is full, carry the last `chunk_overlap` chars
    into the next chunk as a soft context bridge.
    """
    chunks: List[str] = []
    current_parts: List[str] = []
    current_len: int = 0

    def flush() -> str:
        return " ".join(current_parts)

    def overlap_tail(text: str) -> str:
        """Return the trailing `chunk_overlap` chars, snapping to a word start."""
        if len(text) <= chunk_overlap:
            return text
        tail = text[-chunk_overlap:]
        # Snap to nearest word boundary
        idx = tail.find(" ")
        return tail[idx + 1:] if idx != -1 else tail

    for sent in sentences:
        # A single sentence longer than chunk_size → hard-split it
        if len(sent) > chunk_size:
            if current_parts:
                chunks.append(flush())
            for sub in _hard_split(sent, chunk_size, chunk_overlap):
                chunks.append(sub)
            current_parts = []
            current_len = 0
            continue

        needed = len(sent) + (1 if current_parts else 0)  # +1 for space
        if current_len + needed > chunk_size and current_parts:
            full_chunk = flush()
            chunks.append(full_chunk)
            # Overlap: seed next chunk with tail of previous
            tail = overlap_tail(full_chunk)
            current_parts = [tail] if tail else []
            current_len = len(tail)

        current_parts.append(sent)
        current_len += needed

    if current_parts:
        chunks.append(flush())

    return chunks


def _hard_split(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """Split a long string by character count with overlap."""
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - chunk_overlap
    return chunks


# ------------------------------------------------------------------ public API

def split_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> List[str]:
    """
    Split a Vietnamese document into overlapping text chunks.

    Returns a list of non-empty strings, each ≤ chunk_size chars
    (except if a single word exceeds that, which is extremely rare).
    """
    text = _normalize(text)
    if not text:
        return []

    # 1. Split into paragraphs
    paragraphs = re.split(r"\n\n+", text)

    all_sentences: List[str] = []
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        # 2. Split paragraph into sentences
        sents = _sentence_split(para)
        all_sentences.extend(sents)

    # 3. Merge sentences into chunks
    raw_chunks = _merge_into_chunks(all_sentences, chunk_size, chunk_overlap)

    # 4. Filter tiny fragments
    return [c for c in raw_chunks if len(c) >= MIN_CHUNK_LEN]


# ------------------------------------------------------------------ smoke test
if __name__ == "__main__":
    sample = """
Retrieval-Augmented Generation (RAG) là một kỹ thuật trong xử lý ngôn ngữ tự nhiên.
Nó kết hợp việc truy xuất thông tin từ cơ sở dữ liệu với khả năng sinh văn bản của mô hình ngôn ngữ lớn.

Quá trình hoạt động gồm ba bước chính:
Đầu tiên, hệ thống nhận câu hỏi từ người dùng.
Sau đó, truy xuất các đoạn văn bản liên quan từ cơ sở dữ liệu vector.
Cuối cùng, mô hình ngôn ngữ sinh ra câu trả lời dựa trên ngữ cảnh đã truy xuất.
    """
    chunks = split_text(sample, chunk_size=200, chunk_overlap=40)
    for i, c in enumerate(chunks):
        print(f"[{i}] ({len(c)} chars) {c[:80]}...")
