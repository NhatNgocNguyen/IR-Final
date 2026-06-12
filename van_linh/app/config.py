from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "vietnamese-sbert"


@dataclass(frozen=True)
class Settings:
    student_id: str
    embedding_model_path: Path
    teacher_base_url: str
    teacher_proxy_base_url: str
    server_host: str
    server_port: int
    server_public_ip: str | None
    top_k: int
    chunk_size: int
    chunk_overlap: int
    llm_model: str
    llm_timeout_seconds: float

    @property
    def server_url(self) -> str:
        host = self.server_public_ip or self.server_host
        return f"http://{host}:{self.server_port}"


def _get_required(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (PROJECT_ROOT / path).resolve()


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    chunk_size = int(os.getenv("CHUNK_SIZE", "800"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "120"))
    if chunk_overlap >= chunk_size:
        raise RuntimeError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE")

    return Settings(
        student_id=_get_required("STUDENT_ID"),
        embedding_model_path=_resolve_project_path(
            os.getenv(
                "EMBEDDING_MODEL_PATH",
                str(DEFAULT_EMBEDDING_MODEL_PATH.relative_to(PROJECT_ROOT)),
            )
        ),
        teacher_base_url=os.getenv(
            "TEACHER_BASE_URL", "http://192.168.50.218:8000/api/v1"
        ).rstrip("/"),
        teacher_proxy_base_url=os.getenv(
            "TEACHER_PROXY_BASE_URL", "http://192.168.50.218:8000/api/v1/proxy"
        ).rstrip("/"),
        server_host=os.getenv("SERVER_HOST", "0.0.0.0"),
        server_port=int(os.getenv("SERVER_PORT", "8000")),
        server_public_ip=os.getenv("SERVER_PUBLIC_IP"),
        top_k=int(os.getenv("TOP_K", "5")),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        llm_timeout_seconds=float(os.getenv("LLM_TIMEOUT_SECONDS", "30")),
    )
