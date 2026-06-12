"""
main.py — FastAPI Student Server.

Required endpoints (as per exam spec):
  POST /upload   — receive document, chunk + embed + index
  POST /ask      — RAG query → single-letter MCQ answer

Bonus endpoints:
  GET  /health   — status check (vectordb state, chunk count)
"""

import logging
import random
from contextlib import asynccontextmanager
from typing import List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from config import SERVER_HOST, SERVER_PORT, VECTORDB_PATH
from rag_pipeline import answer_question, process_upload
from vectordb import get_db

# ------------------------------------------------------------------ logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
log = logging.getLogger(__name__)


# ================================================================
# Pydantic schemas  (must match teacher's spec exactly)
# ================================================================

class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str

class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str

class AskResponse(BaseModel):
    answer: str                  # MUST be exactly one of: A B C D
    sources: List[str] = []


# ================================================================
# App lifespan — load persisted VectorDB on startup
# ================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("=== Student RAG Server starting ===")

    # Try to restore VectorDB from previous run (saves re-uploading)
    db = get_db()
    if db.load(VECTORDB_PATH):
        log.info("Restored VectorDB: %d chunks (doc_id=%s)", db.n, db.doc_id)
    else:
        log.info("No saved VectorDB found — waiting for /upload")

    # Pre-warm embedding model so first /upload is fast
    try:
        from embedder import embed_texts
        embed_texts(["warmup"])
        log.info("Embedding model warmed up")
    except Exception as exc:
        log.warning("Could not pre-warm embedding model: %s", exc)

    yield   # ← server runs here

    log.info("=== Student RAG Server shutting down ===")


# ================================================================
# App
# ================================================================

app = FastAPI(
    title="Student RAG Server",
    description="LLM + RAG endpoint for PTIT final exam competition",
    version="1.0.0",
    lifespan=lifespan,
)


# ------------------------------------------------------------------ /upload

@app.post("/upload", response_model=UploadResponse)
async def upload(req: UploadRequest):
    """
    Receive a document from the Teacher Server.
    Chunk it, embed it, and store in the in-memory + on-disk VectorDB.
    """
    log.info("/upload  doc_id=%s  text_len=%d", req.doc_id, len(req.text))

    if not req.text or not req.text.strip():
        raise HTTPException(status_code=400, detail="Field 'text' is empty")

    try:
        num_chunks, resolved_id = process_upload(req.text, req.doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    except Exception as exc:
        log.exception("Unexpected error in /upload")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    log.info("/upload  ✓ %d chunks indexed, doc_id=%s", num_chunks, resolved_id)
    return UploadResponse(status="success", doc_id=resolved_id, chunks=num_chunks)


# ------------------------------------------------------------------ /ask

@app.post("/ask", response_model=AskResponse)
async def ask(req: AskRequest):
    """
    Receive a multiple-choice question, retrieve relevant context,
    call the LLM proxy, and return a single-letter answer.
    """
    log.info("/ask  question=%r…", req.question[:80])

    if not req.question or not req.question.strip():
        raise HTTPException(status_code=400, detail="Field 'question' is empty")

    db = get_db()
    if not db.is_ready:
        # Graceful degradation: answer blind (avoid 503 that would score 0)
        log.warning("/ask called but VectorDB not ready — answering without context")

    try:
        answer, sources = answer_question(req.question)
    except Exception as exc:
        log.exception("Unexpected error in /ask")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}")

    # Final guard: ensure the answer is a valid single letter
    if answer not in ("A", "B", "C", "D"):
        invalid = answer
        answer = random.choice(("A", "B", "C", "D"))
        log.error("Invalid answer %r — picking random %s", invalid, answer)

    log.info("/ask  ✓ answer=%s", answer)
    return AskResponse(answer=answer, sources=sources)


# ------------------------------------------------------------------ /health

@app.get("/health")
async def health():
    """Operational status — useful during exam to verify server state."""
    db = get_db()
    return {
        "status":        "ok",
        "vectordb_ready": db.is_ready,
        "chunks":        db.n,
        "doc_id":        db.doc_id,
    }


# ------------------------------------------------------------------ global error handler

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    log.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {type(exc).__name__}"},
    )


# ================================================================
# Entry point
# ================================================================

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
        workers=1,   # single worker — VectorDB lives in memory
        log_level="info",
    )
