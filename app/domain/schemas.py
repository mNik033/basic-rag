from datetime import datetime, timezone
from typing import Any, Optional
from pydantic import BaseModel, Field


class DocumentMetadata(BaseModel):
    filename: str
    content_hash: str
    file_type: str
    file_size: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extra: dict[str, Any] = Field(default_factory=dict)


class ParsedDocument(BaseModel):
    doc_id: str
    text: str
    metadata: DocumentMetadata


class DocumentChunk(BaseModel):
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any] = Field(default_factory=dict)
