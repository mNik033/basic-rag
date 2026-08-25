from datetime import datetime
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RetrievalFilter(BaseModel):
    """Metadata constraints for hybrid search retrieval."""
    repository: Optional[str] = Field(None, description="Full repository name e.g. 'facebook/react'")
    owner: Optional[str] = Field(None, description="Repository owner/organization")
    repo: Optional[str] = Field(None, description="Repository name")
    component: Optional[str] = Field(None, description="Filter by affected component e.g. 'ImageCache'")
    change_type: Optional[str] = Field(None, description="Filter by change type e.g. 'memory', 'performance'")
    author: Optional[str] = Field(None, description="Filter by PR author username")
    architectural_only: bool = Field(False, description="Include only architectural changes")
    breaking_only: bool = Field(False, description="Include only breaking changes")
    milestone: Optional[str] = Field(None, description="Filter by release milestone")
    since: Optional[datetime] = Field(None, description="Filter PRs merged on or after this timestamp")
    until: Optional[datetime] = Field(None, description="Filter PRs merged on or before this timestamp")


class RetrievalCandidate(BaseModel):
    """A scored and ranked Pull Request candidate retrieved from hybrid search."""
    pr_number: int
    repository: str
    title: str
    author: str
    state: str
    merged_at: Optional[datetime] = None
    milestone: Optional[str] = None
    summary: Optional[str] = None
    motivation_reason: Optional[str] = None
    motivation_type: Optional[str] = None
    components: List[str] = Field(default_factory=list)
    change_types: List[str] = Field(default_factory=list)
    architectural_change: bool = False
    breaking_change: bool = False
    key_technical_details: List[str] = Field(default_factory=list)
    changed_files: List[str] = Field(default_factory=list)
    vector_score: Optional[float] = Field(None, description="Cosine similarity score (0.0 - 1.0)")
    keyword_score: Optional[float] = Field(None, description="Lexical matching relevance score")
    combined_score: float = Field(0.0, description="Fused RRF / weighted hybrid score")
    rank: int = Field(0, description="Final rank position (1-indexed)")
    match_reasons: List[str] = Field(default_factory=list, description="Explanations for why this PR matched")


class HybridSearchRequest(BaseModel):
    """Input payload for hybrid engineering search."""
    query: str = Field(..., min_length=1, description="Natural language search query or engineering question")
    filter: Optional[RetrievalFilter] = Field(default_factory=RetrievalFilter, description="Metadata filters")
    limit: int = Field(10, ge=1, le=50, description="Max number of ranked PR candidates to return")
    vector_weight: float = Field(0.6, ge=0.0, le=1.0, description="Weight multiplier for dense vector retrieval")
    keyword_weight: float = Field(0.4, ge=0.0, le=1.0, description="Weight multiplier for lexical keyword search")
    rrf_k: int = Field(60, ge=1, le=100, description="Smoothing constant k for Reciprocal Rank Fusion")


class HybridSearchResponse(BaseModel):
    """Output results of hybrid search with ranked evidence candidates."""
    query: str
    total_candidates: int
    results: List[RetrievalCandidate]
    vector_hits: int = 0
    keyword_hits: int = 0


class EngineeringEvidence(BaseModel):
    """Specific PR evidence citation backing an answer."""
    pr_number: int
    repository: str
    title: str
    author: str
    merged_at: Optional[datetime] = None
    milestone: Optional[str] = None
    components: List[str] = Field(default_factory=list)
    change_types: List[str] = Field(default_factory=list)
    motivation_type: Optional[str] = Field(None, description="'documented', 'inferred', or 'unknown'")
    motivation_reason: Optional[str] = None
    key_technical_details: List[str] = Field(default_factory=list)
    changed_files: List[str] = Field(default_factory=list)
    relevance_score: float = Field(0.0, description="Rank fusion relevance score")
    rank: int = 0
    match_reasons: List[str] = Field(default_factory=list)


class EngineeringQueryRequest(BaseModel):
    """Input payload for asking natural language questions about engineering evolution."""
    query: str = Field(..., min_length=2, description="Natural language engineering question")
    repository: Optional[str] = Field(None, description="Scope query to a specific repo (e.g. 'facebook/react')")
    filter: Optional[RetrievalFilter] = Field(default_factory=RetrievalFilter, description="Optional metadata constraints")
    limit: int = Field(5, ge=1, le=20, description="Number of top PR candidates to retrieve as context")
    system_prompt: Optional[str] = Field(None, description="Custom system prompt override")
    include_raw_evidence: bool = Field(True, description="Whether to include full structured evidence list in response")


class EngineeringAnswerResponse(BaseModel):
    """Evidence-backed engineering intelligence answer."""
    query: str
    answer: str
    scenario_detected: str = Field("general_qa", description="Detected question category (e.g. release_comparison, issue_search, impact_search, decision_understanding)")
    evidence: List[EngineeringEvidence] = Field(default_factory=list)
    total_evidence_count: int = 0
    has_sufficient_evidence: bool = True
    model_used: str

