from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory, get_db
from app.domain.github import (
    GitHubSyncRequest,
    GitHubSyncResponse,
    GitHubSyncStatusResponse,
)
from app.domain.models import ChangedFile, Commit, PullRequest, Repository, SyncState
from app.services.github.collector import GitHubCollectorService

router = APIRouter(prefix="/github", tags=["GitHub Engineering Data"])


async def _run_sync_task(request: GitHubSyncRequest) -> None:
    """Background task worker for repository synchronization."""
    async with async_session_factory() as session:
        try:
            collector = GitHubCollectorService(session)
            await collector.sync_repository(request)
        except Exception:
            pass


@router.post("/sync", response_model=GitHubSyncResponse, status_code=status.HTTP_200_OK)
async def trigger_sync(
    request: GitHubSyncRequest,
    background_tasks: BackgroundTasks,
    background: bool = Query(
        False, description="Run sync in background and return immediately"
    ),
    db: AsyncSession = Depends(get_db),
) -> GitHubSyncResponse:
    """Sync Pull Requests, commits, diffs, and reviews for a GitHub repository."""
    if background:
        background_tasks.add_task(_run_sync_task, request)
        return GitHubSyncResponse(
            repository=f"{request.owner}/{request.repo}",
            status="pending",
            message="GitHub synchronization triggered in background.",
        )

    collector = GitHubCollectorService(db)
    return await collector.sync_repository(request)


@router.get(
    "/sync/status/{owner}/{repo}",
    response_model=GitHubSyncStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_sync_status(
    owner: str,
    repo: str,
    db: AsyncSession = Depends(get_db),
) -> GitHubSyncStatusResponse:
    """Get the current sync status for a repository."""
    stmt = (
        select(SyncState)
        .join(Repository)
        .where(Repository.owner == owner, Repository.name == repo)
    )
    result = await db.execute(stmt)
    state = result.scalar_one_or_none()

    if not state:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Repository {owner}/{repo} has not been synced yet.",
        )

    return GitHubSyncStatusResponse(
        repository=f"{owner}/{repo}",
        status=state.status,
        last_synced_pr_number=state.last_synced_pr_number,
        total_prs_synced=state.total_prs_synced,
        last_synced_at=state.last_synced_at,
        error_message=state.error_message,
    )


@router.get("/repositories", response_model=List[Dict[str, Any]])
async def list_repositories(
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List all tracked repositories along with total PR counts."""
    stmt = (
        select(
            Repository.id,
            Repository.owner,
            Repository.name,
            Repository.default_branch,
            Repository.description,
            Repository.created_at,
            func.count(PullRequest.id).label("total_prs"),
        )
        .outerjoin(PullRequest, PullRequest.repository_id == Repository.id)
        .group_by(Repository.id)
    )
    result = await db.execute(stmt)
    rows = result.all()

    return [
        {
            "id": r.id,
            "owner": r.owner,
            "name": r.name,
            "full_name": f"{r.owner}/{r.name}",
            "default_branch": r.default_branch,
            "description": r.description,
            "created_at": r.created_at,
            "total_prs": r.total_prs,
        }
        for r in rows
    ]


@router.get("/pull-requests/{owner}/{repo}", response_model=List[Dict[str, Any]])
async def list_pull_requests(
    owner: str,
    repo: str,
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List ingested pull requests with associated commit and changed file counts."""
    repo_stmt = select(Repository).where(Repository.owner == owner, Repository.name == repo)
    repo_res = await db.execute(repo_stmt)
    repo_obj = repo_res.scalar_one_or_none()

    if not repo_obj:
        raise HTTPException(status_code=404, detail=f"Repository {owner}/{repo} not found.")

    pr_stmt = (
        select(
            PullRequest.id,
            PullRequest.number,
            PullRequest.title,
            PullRequest.author,
            PullRequest.state,
            PullRequest.merged_at,
            PullRequest.labels,
            PullRequest.milestone,
            func.count(func.distinct(Commit.id)).label("commit_count"),
            func.count(func.distinct(ChangedFile.id)).label("file_count"),
        )
        .outerjoin(Commit, Commit.pull_request_id == PullRequest.id)
        .outerjoin(ChangedFile, ChangedFile.pull_request_id == PullRequest.id)
        .where(PullRequest.repository_id == repo_obj.id)
        .group_by(PullRequest.id)
        .order_by(PullRequest.merged_at.desc().nullslast(), PullRequest.number.desc())
        .limit(limit)
        .offset(offset)
    )
    result = await db.execute(pr_stmt)
    rows = result.all()

    return [
        {
            "id": r.id,
            "number": r.number,
            "title": r.title,
            "author": r.author,
            "state": r.state,
            "merged_at": r.merged_at,
            "labels": r.labels,
            "milestone": r.milestone,
            "commits": r.commit_count,
            "changed_files": r.file_count,
        }
        for r in rows
    ]
