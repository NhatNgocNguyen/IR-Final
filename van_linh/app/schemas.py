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
    status: str | None = None
    message: str | None = None
    student_id: str | None = None
    server_url: str | None = None


class EvaluateResponse(BaseModel):
    student_id: str | None = None
    score: float | None = None
    status: str | None = None
    detail: list[dict] | list[str] | None = None


class ResetResponse(BaseModel):
    status: str
    message: str


class ResultResponse(BaseModel):
    student_id: str | None = None
    score: float | None = None
    status: str | None = None
    current_question: int | None = None
