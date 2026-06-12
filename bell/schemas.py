from pydantic import BaseModel, Field
from typing import Optional, List


class UploadRequest(BaseModel):
    doc_id: Optional[str] = None
    text: str


class UploadResponse(BaseModel):
    status: str
    doc_id: Optional[str] = None
    chunks: int


class AskRequest(BaseModel):
    id: Optional[int] = None
    question: str
    note: Optional[str] = None
    options: Optional[List[str]] = None


class AskResponse(BaseModel):
    answer: str
    sources: List[str] = Field(default_factory=list)
