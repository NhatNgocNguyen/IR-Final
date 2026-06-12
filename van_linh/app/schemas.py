from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field, field_validator


class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str = Field(..., min_length=1)

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("text must not be empty")
        return cleaned


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class AskResponse(BaseModel):
    answer: str = Field(..., pattern="^[ABCD]$")
    sources: list[str] = Field(default_factory=list)


class RegisterPayload(BaseModel):
    server_url: str


class RegisterResponse(BaseModel):
    message: str
    student_id: str
    server_url: str


class EvaluateResponse(BaseModel):
    message: str
    final_score: float


class EvaluateRequest(BaseModel):
    document_received: bool = False


class ResetResponse(BaseModel):
    status: str
    message: str
    score: float


class ResultResponse(BaseModel):
    student_id: str
    score: float
    status: str
    current_question: int
