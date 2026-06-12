"""
rag_pipeline.py — Full RAG orchestration.

Two public functions:
  process_upload(text, doc_id)  →  (num_chunks, resolved_doc_id)
  answer_question(question)     →  (answer_letter, source_chunks)
"""

import logging
import re
from typing import List, Optional, Tuple

from chunker import split_text
from config import MAX_CONTEXT_CHARS, TOP_K_FINAL
from embedder import embed_texts
from llm_client import chat
from retriever import build_context, retrieve_chunks
from vectordb import get_db

log = logging.getLogger(__name__)


# ================================================================
# SYSTEM PROMPT (Vietnamese — tight, no fluff)
# ================================================================

_SYSTEM = (
    "Bạn là chuyên gia trả lời câu hỏi trắc nghiệm dựa trên tài liệu.\n"
    "QUY TẮC BẮT BUỘC: Chỉ trả lời bằng ĐÚNG MỘT ký tự: A, B, C hoặc D.\n"
    "Tuyệt đối không thêm bất kỳ chữ nào khác."
)


# ================================================================
# Upload processing
# ================================================================

def process_upload(
    text: str,
    doc_id: Optional[str] = None,
) -> Tuple[int, str]:
    """
    Chunk → embed → index in VectorDB → persist.

    Returns (num_chunks, doc_id).
    Raises ValueError if nothing usable could be extracted.
    """
    resolved_id = doc_id or "uploaded_doc"
    log.info("process_upload: doc_id=%s, text_len=%d", resolved_id, len(text))

    chunks = split_text(text)
    if not chunks:
        raise ValueError("No valid chunks could be extracted from the document.")

    log.info("Chunks: %d  (avg %d chars each)",
             len(chunks), sum(len(c) for c in chunks) // len(chunks))

    log.info("Embedding %d chunks…", len(chunks))
    embeddings = embed_texts(chunks)
    log.info("Embedding done — shape %s", embeddings.shape)

    db = get_db()
    db.add_documents(chunks, embeddings, resolved_id)
    db.save()

    return len(chunks), resolved_id


# ================================================================
# Question answering
# ================================================================

def answer_question(question: str) -> Tuple[str, List[str]]:
    """
    Full RAG pipeline for a single multiple-choice question.

    Returns:
        answer  — one of "A" "B" "C" "D"
        sources — list of retrieved chunk strings (for the response body)
    """
    db = get_db()

    # ── Retrieve ────────────────────────────────────────────────
    chunks = retrieve_chunks(question, db=db, top_k=TOP_K_FINAL)

    if not chunks:
        log.warning("No chunks retrieved — answering blind")
        letter = _ask_llm(question, context=None)
        return letter, []

    # ── Assemble context within char budget ─────────────────────
    context = build_context(chunks, max_chars=MAX_CONTEXT_CHARS)

    # ── Build prompt ─────────────────────────────────────────────
    user_msg = _build_user_message(question, context)

    # Safety check: log total prompt size
    total = len(_SYSTEM) + len(user_msg)
    log.debug("Prompt size: %d chars (system=%d, user=%d)",
              total, len(_SYSTEM), len(user_msg))

    # ── Call LLM ────────────────────────────────────────────────
    raw = chat(
        messages=[
            {"role": "system",  "content": _SYSTEM},
            {"role": "user",    "content": user_msg},
        ],
        max_tokens=8,
        temperature=0.0,
    )

    letter = _extract_answer(raw)
    log.info("Q: %s…  raw=%r  answer=%s", question[:60], raw, letter)

    # Return top-3 chunk snippets as sources (truncated)
    sources = [c[:200] for c in chunks[:3]]
    return letter, sources


# ================================================================
# Private helpers
# ================================================================

def _build_user_message(question: str, context: Optional[str]) -> str:
    if context:
        return (
            "Dựa vào tài liệu dưới đây, hãy chọn đáp án đúng nhất cho câu hỏi trắc nghiệm.\n\n"
            "=== TÀI LIỆU ===\n"
            f"{context}\n"
            "=== HẾT TÀI LIỆU ===\n\n"
            f"Câu hỏi: {question}\n\n"
            "Trả lời (chỉ 1 ký tự A/B/C/D):"
        )
    else:
        return (
            f"Câu hỏi: {question}\n\n"
            "Trả lời (chỉ 1 ký tự A/B/C/D):"
        )


def _ask_llm(question: str, context: Optional[str]) -> str:
    """Low-level LLM call, returns extracted letter."""
    raw = chat(
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user",   "content": _build_user_message(question, context)},
        ],
        max_tokens=8,
        temperature=0.0,
    )
    return _extract_answer(raw)


def _extract_answer(text: str) -> str:
    """
    Robustly extract A/B/C/D from LLM output.
    Handles: 'B', 'B.', 'Đáp án: C', 'Câu trả lời là D', 'The answer is A', etc.
    """
    if not text:
        return "A"

    t = text.strip().upper()

    # 1. Direct single character
    if t in ("A", "B", "C", "D"):
        return t

    # 2. Answer pattern: "Đáp án: B", "Answer: C", "is D"
    m = re.search(r"(?:ĐÁP ÁN|ANSWER|ANS|IS)[:\s]+([ABCD])", t)
    if m:
        return m.group(1)

    # 3. Parenthesised: "(B)" or "[C]"
    m = re.search(r"[(\[]\s*([ABCD])\s*[)\]]", t)
    if m:
        return m.group(1)

    # 4. "B." or "B)"
    m = re.search(r"\b([ABCD])[.)]", t)
    if m:
        return m.group(1)

    # 5. First standalone letter in the string
    m = re.search(r"\b([ABCD])\b", t)
    if m:
        return m.group(1)

    # 6. Any occurrence of A/B/C/D (last resort)
    m = re.search(r"[ABCD]", t)
    if m:
        return m.group(0)

    log.warning("Cannot extract answer from %r — defaulting to A", text)
    return "A"
