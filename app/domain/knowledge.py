from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class EngineeringDocument(BaseModel):
    """Searchable composite engineering document representing a Pull Request and its structured knowledge."""
    doc_id: str = Field(..., description="Unique document ID (e.g. 'gh_pr_owner_repo_16210')")
    text: str = Field(..., description="Rich searchable text synthesis")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for hybrid filtering")


class KnowledgeIndexRequest(BaseModel):
    repository_id: Optional[int] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    pr_number: Optional[int] = None
    limit: Optional[int] = Field(100, ge=1, le=1000, description="Max PRs to index into vector store")
    force_reindex: bool = Field(False, description="Re-generate and overwrite existing vector embeddings")


class KnowledgeIndexResponse(BaseModel):
    repository: str
    status: str
    documents_indexed: int
    chunks_created: int
    message: str


class KnowledgeStatusResponse(BaseModel):
    repository: str
    total_prs: int
    understood_prs: int
    indexed_vectors: int
