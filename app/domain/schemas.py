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


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: dict[str, Any]
    similarity_score: Optional[float] = None


class RAGQueryResult(BaseModel):
    query: str
    answer: str
    sources: list[RetrievedChunk]
    model: str
    cached: bool = False


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="The question or prompt to ask.")
    n_results: int = Field(default=3, ge=1, le=10, description="Number of context chunks to retrieve.")
    system_prompt: Optional[str] = Field(default=None, description="Optional custom system prompt override.")
    use_cache: bool = Field(default=True, description="Whether to check and populate the semantic query cache.")
    similarity_threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Minimum cosine similarity score (0.0 - 1.0) required to include a context chunk.",
    )


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Search term or phrase.")
    n_results: int = Field(default=3, ge=1, le=20, description="Maximum number of chunks to return.")


class SearchResponse(BaseModel):
    query: str
    total_results: int
    results: list[RetrievedChunk]
