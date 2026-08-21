from datetime import datetime
from typing import Any, List, Optional
from pydantic import BaseModel, Field


class GitHubSyncRequest(BaseModel):
    owner: str = Field(..., description="Repository owner / organization", min_length=1)
    repo: str = Field(..., description="Repository name", min_length=1)
    token: Optional[str] = Field(None, description="Optional per-request GitHub PAT")
    limit: Optional[int] = Field(None, description="Max number of PRs to sync (default: all)", ge=1)
    since: Optional[datetime] = Field(None, description="Only sync PRs merged/updated since this timestamp")
    force_resync: bool = Field(False, description="Re-ingest PRs even if already synced")


class GitHubSyncResponse(BaseModel):
    repository: str
    status: str
    message: str
    total_synced: int = 0
    prs_processed: int = 0
    commits_processed: int = 0
    files_processed: int = 0
    reviews_processed: int = 0


class GitHubSyncStatusResponse(BaseModel):
    repository: str
    status: str
    last_synced_pr_number: Optional[int] = None
    total_prs_synced: int = 0
    last_synced_at: Optional[datetime] = None
    error_message: Optional[str] = None


class RawCommit(BaseModel):
    sha: str
    author_name: Optional[str] = None
    author_email: Optional[str] = None
    message: str
    committed_at: Optional[datetime] = None
    additions: int = 0
    deletions: int = 0


class RawChangedFile(BaseModel):
    filename: str
    status: str
    additions: int = 0
    deletions: int = 0
    changes: int = 0
    patch_text: Optional[str] = None


class RawReviewComment(BaseModel):
    github_comment_id: int
    review_id: Optional[int] = None
    author: str
    path: Optional[str] = None
    line: Optional[int] = None
    body: str
    created_at: datetime


class RawReview(BaseModel):
    github_review_id: int
    author: str
    state: str
    body: Optional[str] = None
    submitted_at: Optional[datetime] = None
    comments: List[RawReviewComment] = Field(default_factory=list)


class RawPullRequest(BaseModel):
    github_pr_id: int
    number: int
    title: str
    body: Optional[str] = None
    state: str
    author: str
    merged_at: Optional[datetime] = None
    merge_commit_sha: Optional[str] = None
    labels: List[str] = Field(default_factory=list)
    milestone: Optional[str] = None
    created_at: datetime
    closed_at: Optional[datetime] = None
    commits: List[RawCommit] = Field(default_factory=list)
    changed_files: List[RawChangedFile] = Field(default_factory=list)
    reviews: List[RawReview] = Field(default_factory=list)
    review_comments: List[RawReviewComment] = Field(default_factory=list)
    raw_json: Optional[dict[str, Any]] = None
