from enum import Enum
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class EvidenceType(str, Enum):
    DOCUMENTED = "documented"  # Explicitly stated in PR body, commit message, or review comments
    INFERRED = "inferred"      # Deduced from code diffs and file changes
    UNKNOWN = "unknown"        # Insufficient evidence to determine reason


class MotivationDetail(BaseModel):
    evidence_type: EvidenceType = Field(
        ...,
        description="Whether the motivation is explicitly documented, inferred from code, or unknown",
    )
    reason: str = Field(
        ...,
        description="Why the change was made (the problem being solved or goal achieved)",
    )
    evidence_quote: Optional[str] = Field(
        None,
        description="Direct quote from PR description or review discussion if documented",
    )


class PRUnderstandingResult(BaseModel):
    summary: str = Field(
        ...,
        description="Concise 1-2 sentence engineering summary of what changed",
    )
    motivation: MotivationDetail = Field(
        ...,
        description="Problem or reason behind the change, classified as documented, inferred, or unknown",
    )
    components: List[str] = Field(
        default_factory=list,
        description="Names of impacted modules, subsystems, packages, or major classes",
    )
    change_types: List[str] = Field(
        default_factory=list,
        description="List of change categories: e.g. memory, performance, bugfix, refactor, feature, security, api-change, build-infra",
    )
    impact: List[str] = Field(
        default_factory=list,
        description="Concrete areas affected: e.g. 'reduced memory allocation in cache', 'faster page load'",
    )
    architectural_change: bool = Field(
        False,
        description="True if the PR introduces/alters architecture, design patterns, or major interfaces",
    )
    breaking_change: bool = Field(
        False,
        description="True if this change is backward-incompatible for callers or public APIs",
    )
    key_technical_details: List[str] = Field(
        default_factory=list,
        description="Key technical notes, algorithms, data structures, or parameters modified",
    )
    raw_response: Optional[dict[str, Any]] = None


class PRUnderstandingProcessRequest(BaseModel):
    repository_id: Optional[int] = None
    owner: Optional[str] = None
    repo: Optional[str] = None
    pr_number: Optional[int] = None
    limit: Optional[int] = Field(50, ge=1, le=500, description="Max PRs to analyze in this batch")
    force_reprocess: bool = Field(False, description="Re-analyze even if already summarized")


class PRUnderstandingProcessResponse(BaseModel):
    repository: str
    status: str
    processed_count: int
    failed_count: int
    message: str
