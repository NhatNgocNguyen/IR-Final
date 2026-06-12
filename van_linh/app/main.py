from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from app.config import get_settings
from app.llm import LlmService, TeacherProxyRequestError, TeacherProxyTimeoutError
from app.rag import RagService
from app.schemas import AskRequest, AskResponse, UploadRequest, UploadResponse


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    LOGGER.info("Starting Student Server on %s:%s", settings.server_host, settings.server_port)
    app.state.settings = settings
    app.state.rag_service = RagService(settings)
    app.state.llm_service = LlmService(settings)
    yield


app = FastAPI(title="Student RAG Server", lifespan=lifespan)


@app.get("/health")
def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/upload", response_model=UploadResponse)
def upload_document(payload: UploadRequest) -> UploadResponse:
    LOGGER.info("Received /upload request for doc_id=%s", payload.doc_id)
    rag_service: RagService = app.state.rag_service
    try:
        doc_id, chunk_count = rag_service.ingest(payload.text, payload.doc_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    LOGGER.info("Indexed %s chunks for doc_id=%s", chunk_count, doc_id)
    return UploadResponse(status="success", doc_id=doc_id, chunks=chunk_count)


@app.post("/ask", response_model=AskResponse)
def ask_question(payload: AskRequest) -> AskResponse:
    LOGGER.info("Received /ask request")
    rag_service: RagService = app.state.rag_service
    llm_service: LlmService = app.state.llm_service

    try:
        retrieval = rag_service.retrieve(payload.question, app.state.settings.top_k)
    except LookupError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    try:
        answer = llm_service.answer_question(payload.question, retrieval.chunks)
    except TeacherProxyTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except TeacherProxyRequestError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    LOGGER.info("Returning answer=%s for /ask", answer)
    return AskResponse(answer=answer, sources=retrieval.chunks)
