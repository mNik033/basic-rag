from datetime import datetime, timezone
from typing import Any, List, Optional
from sqlalchemy import (
    BigInteger,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.types import JSON
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.database import Base


# Cross-dialect JSON type (falls back to generic JSON on non-Postgres engines)
JSONType = JSON().with_variant(JSONB, "postgresql")


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Repository(Base):
    __tablename__ = "github_repositories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    owner: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    default_branch: Mapped[str] = mapped_column(String(100), default="main", nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )

    # Relationships
    pull_requests: Mapped[List["PullRequest"]] = relationship(
        "PullRequest", back_populates="repository", cascade="all, delete-orphan"
    )
    sync_state: Mapped[Optional["SyncState"]] = relationship(
        "SyncState", back_populates="repository", uselist=False, cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("owner", "name", name="uq_repo_owner_name"),
    )


class PullRequest(Base):
    __tablename__ = "github_pull_requests"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_repositories.id", ondelete="CASCADE"), nullable=False, index=True
    )
    github_pr_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    state: Mapped[str] = mapped_column(String(50), nullable=False, default="closed")
    author: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    merged_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True, index=True
    )
    merge_commit_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    labels: Mapped[list[str]] = mapped_column(JSONType, default=list, nullable=False)
    milestone: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSONType, nullable=True)

    # Relationships
    repository: Mapped["Repository"] = relationship("Repository", back_populates="pull_requests")
    commits: Mapped[List["Commit"]] = relationship(
        "Commit", back_populates="pull_request", cascade="all, delete-orphan"
    )
    changed_files: Mapped[List["ChangedFile"]] = relationship(
        "ChangedFile", back_populates="pull_request", cascade="all, delete-orphan"
    )
    reviews: Mapped[List["Review"]] = relationship(
        "Review", back_populates="pull_request", cascade="all, delete-orphan"
    )
    review_comments: Mapped[List["ReviewComment"]] = relationship(
        "ReviewComment", back_populates="pull_request", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("repository_id", "number", name="uq_repo_pr_number"),
        Index("idx_pr_repo_merged", "repository_id", "merged_at"),
    )


class Commit(Base):
    __tablename__ = "github_commits"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pull_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_pull_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    sha: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    author_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    author_email: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    message: Mapped[str] = mapped_column(Text, nullable=False)
    committed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    additions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deletions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    pull_request: Mapped["PullRequest"] = relationship("PullRequest", back_populates="commits")

    __table_args__ = (
        UniqueConstraint("pull_request_id", "sha", name="uq_pr_commit_sha"),
    )


class ChangedFile(Base):
    __tablename__ = "github_changed_files"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pull_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_pull_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    filename: Mapped[str] = mapped_column(String(1000), nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="modified")
    additions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    deletions: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    changes: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    patch_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    pull_request: Mapped["PullRequest"] = relationship("PullRequest", back_populates="changed_files")


class Review(Base):
    __tablename__ = "github_reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pull_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_pull_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    github_review_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    author: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    state: Mapped[str] = mapped_column(String(50), nullable=False)
    body: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    submitted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    pull_request: Mapped["PullRequest"] = relationship("PullRequest", back_populates="reviews")
    comments: Mapped[List["ReviewComment"]] = relationship(
        "ReviewComment", back_populates="review", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("pull_request_id", "github_review_id", name="uq_pr_review_id"),
    )


class ReviewComment(Base):
    __tablename__ = "github_review_comments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    pull_request_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_pull_requests.id", ondelete="CASCADE"), nullable=False, index=True
    )
    review_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("github_reviews.id", ondelete="SET NULL"), nullable=True, index=True
    )
    github_comment_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    author: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    path: Mapped[Optional[str]] = mapped_column(String(1000), nullable=True)
    line: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    body: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)

    pull_request: Mapped["PullRequest"] = relationship("PullRequest", back_populates="review_comments")
    review: Mapped[Optional["Review"]] = relationship("Review", back_populates="comments")


class SyncState(Base):
    __tablename__ = "github_sync_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    repository_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("github_repositories.id", ondelete="CASCADE"), nullable=False, unique=True
    )
    last_synced_pr_number: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    total_prs_synced: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_synced_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(50), default="idle", nullable=False)
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    repository: Mapped["Repository"] = relationship("Repository", back_populates="sync_state")
