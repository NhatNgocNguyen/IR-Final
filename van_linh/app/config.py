from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_EMBEDDING_MODEL_PATH = PROJECT_ROOT / "models" / "vietnamese-sbert"
DEFAULT_EMBEDDING_MODEL_NAME = "keepitreal/vietnamese-sbert"


@dataclass(frozen=True)
class Settings:
    student_id: str
    embedding_model_path: Path
    embedding_model_name: str
    index_storage_dir: Path
    teacher_base_url: str
    teacher_proxy_base_url: str
    server_host: str
    server_port: int
    server_public_ip: str | None
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_chunk_tokens: int
    llm_model: str
    prompt_variant: int
    max_prompt_tokens: int
    prompt_token_margin: int
    gpt_chars_per_token: float
    llm_max_retries: int
    llm_timeout_seconds: float
    evaluate_timeout_seconds: float
    mmr_lambda: float
    dedup_threshold: float
    neighbor_radius: int

    @property
    def server_url(self) -> str:
        host = self.server_public_ip or self.server_host
        return f"http://{host}:{self.server_port}"

    @property
    def embedding_model_local_only(self) -> bool:
        """True when the project ships a local model folder; False uses HF cache/id."""
        return self.embedding_model_path.exists()

    @property
    def embedding_model_source(self) -> str:
        """Local folder path if present, otherwise the Hugging Face model id."""
        if self.embedding_model_local_only:
            return str(self.embedding_model_path)
        return self.embedding_model_name


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


def _float_or_default(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() == "auto":
        return default
    return float(raw)


def _int_or_default(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip().lower() == "auto":
        return default
    return int(raw)


def _prompt_variant_or_default(default: int = 2, *, min_value: int = 1, max_value: int = 4) -> int:
    variant = _int_or_default("PROMPT_VARIANT", default)
    if variant < min_value or variant > max_value:
        return default
    return variant


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
        embedding_model_name=os.getenv(
            "EMBEDDING_MODEL_NAME", DEFAULT_EMBEDDING_MODEL_NAME
        ),
        index_storage_dir=_resolve_project_path(
            os.getenv("INDEX_STORAGE_DIR", "storage/index")
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
        max_chunk_tokens=_int_or_default("MAX_CHUNK_TOKENS", 250),
        llm_model=os.getenv("LLM_MODEL", "gpt-4o-mini"),
        prompt_variant=_prompt_variant_or_default(),
        max_prompt_tokens=_int_or_default("MAX_PROMPT_TOKENS", 4000),
        prompt_token_margin=_int_or_default("PROMPT_TOKEN_MARGIN", 300),
        gpt_chars_per_token=_float_or_default("GPT_CHARS_PER_TOKEN", 2.5),
        llm_max_retries=_int_or_default("LLM_MAX_RETRIES", 1),
        llm_timeout_seconds=_float_or_default("LLM_TIMEOUT_SECONDS", 45.0),
        evaluate_timeout_seconds=float(os.getenv("EVALUATE_TIMEOUT_SECONDS", "7200")),
        mmr_lambda=_float_or_default("MMR_LAMBDA", 0.7),
        dedup_threshold=float(os.getenv("DEDUP_THRESHOLD", "0.97")),
        neighbor_radius=_int_or_default("NEIGHBOR_RADIUS", 1),
    )
