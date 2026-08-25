from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import async_session_factory, get_db
from app.domain.github import (
    GitHubSyncRequest,
    GitHubSyncResponse,
    GitHubSyncStatusResponse,
)
from app.domain.knowledge import (
    KnowledgeIndexRequest,
    KnowledgeIndexResponse,
    KnowledgeStatusResponse,
)
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository, SyncState
from app.domain.retrieval import (
    EngineeringAnswerResponse,
    EngineeringQueryRequest,
    HybridSearchRequest,
    HybridSearchResponse,
)
from app.domain.understanding import (
    PRUnderstandingProcessRequest,
    PRUnderstandingProcessResponse,
)
from app.services.github.collector import GitHubCollectorService
from app.services.github.knowledge_service import KnowledgeBaseService
from app.services.github.rag_service import EngineeringRAGService
from app.services.github.understanding_service import PRUnderstandingService
from app.services.retrieval.hybrid_search import HybridSearchEngine

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


@router.post(
    "/understanding/process",
    response_model=PRUnderstandingProcessResponse,
    status_code=status.HTTP_200_OK,
)
async def process_pr_understanding(
    request: PRUnderstandingProcessRequest,
    db: AsyncSession = Depends(get_db),
) -> PRUnderstandingProcessResponse:
    """Analyze ingested PRs with LLM to generate structured engineering summaries."""
    service = PRUnderstandingService(db)
    return await service.process_batch(request)


@router.get("/understanding/{owner}/{repo}/{pr_number}", response_model=Dict[str, Any])
async def get_pr_understanding(
    owner: str,
    repo: str,
    pr_number: int,
    db: AsyncSession = Depends(get_db),
) -> Dict[str, Any]:
    """Retrieve structured engineering understanding for a single PR."""
    stmt = (
        select(PRUnderstanding, PullRequest)
        .join(PullRequest, PRUnderstanding.pull_request_id == PullRequest.id)
        .join(Repository, PullRequest.repository_id == Repository.id)
        .where(
            Repository.owner == owner,
            Repository.name == repo,
            PullRequest.number == pr_number,
        )
    )
    res = await db.execute(stmt)
    row = res.first()

    if not row:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"PR #{pr_number} for {owner}/{repo} has not been analyzed yet.",
        )

    u, pr = row
    return {
        "repository": f"{owner}/{repo}",
        "pr_number": pr.number,
        "pr_title": pr.title,
        "author": pr.author,
        "merged_at": pr.merged_at,
        "summary": u.summary,
        "motivation": {
            "evidence_type": u.motivation_type,
            "reason": u.motivation_reason,
            "evidence_quote": u.motivation_quote,
        },
        "components": u.components,
        "change_types": u.change_types,
        "impact": u.impact,
        "architectural_change": u.architectural_change,
        "breaking_change": u.breaking_change,
        "key_technical_details": u.key_technical_details,
        "model_used": u.model_used,
        "updated_at": u.updated_at,
    }


@router.get("/understanding/{owner}/{repo}", response_model=List[Dict[str, Any]])
async def list_pr_understandings(
    owner: str,
    repo: str,
    change_type: Optional[str] = Query(None, description="Filter by change type (e.g. memory, performance)"),
    architectural_only: bool = Query(False, description="Filter only architectural changes"),
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
) -> List[Dict[str, Any]]:
    """List structured engineering knowledge entries for a repository."""
    stmt = (
        select(PRUnderstanding, PullRequest)
        .join(PullRequest, PRUnderstanding.pull_request_id == PullRequest.id)
        .join(Repository, PullRequest.repository_id == Repository.id)
        .where(Repository.owner == owner, Repository.name == repo)
    )

    if architectural_only:
        stmt = stmt.where(PRUnderstanding.architectural_change == True)  # noqa: E712

    stmt = stmt.order_by(PullRequest.number.desc()).limit(limit).offset(offset)
    res = await db.execute(stmt)
    rows = res.all()

    results = []
    for u, pr in rows:
        if change_type and change_type not in u.change_types:
            continue
        results.append(
            {
                "pr_number": pr.number,
                "pr_title": pr.title,
                "author": pr.author,
                "merged_at": pr.merged_at,
                "summary": u.summary,
                "motivation": {
                    "evidence_type": u.motivation_type,
                    "reason": u.motivation_reason,
                    "evidence_quote": u.motivation_quote,
                },
                "components": u.components,
                "change_types": u.change_types,
                "impact": u.impact,
                "architectural_change": u.architectural_change,
                "breaking_change": u.breaking_change,
                "key_technical_details": u.key_technical_details,
            }
        )

    return results


@router.post(
    "/knowledge/index",
    response_model=KnowledgeIndexResponse,
    status_code=status.HTTP_200_OK,
)
async def index_repository_knowledge(
    request: KnowledgeIndexRequest,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeIndexResponse:
    """Trigger vector indexing and dense embedding generation for a repository's pull requests."""
    service = KnowledgeBaseService(db)
    try:
        return await service.index_repository_knowledge(request)
    except ValueError as e:
        status_code = (
            status.HTTP_404_NOT_FOUND
            if "not found" in str(e).lower()
            else status.HTTP_400_BAD_REQUEST
        )
        raise HTTPException(status_code=status_code, detail=str(e))


@router.get(
    "/knowledge/status/{owner}/{repo}",
    response_model=KnowledgeStatusResponse,
    status_code=status.HTTP_200_OK,
)
async def get_knowledge_status(
    owner: str,
    repo: str,
    db: AsyncSession = Depends(get_db),
) -> KnowledgeStatusResponse:
    """Get count of total PRs, understood PRs, and indexed vector embeddings for a repository."""
    service = KnowledgeBaseService(db)
    try:
        return await service.get_knowledge_status(owner=owner, repo=repo)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e),
        )


@router.post(
    "/search/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (semantic + lexical + RRF) over repository pull requests",
)
async def hybrid_search_pr_knowledge(
    request: HybridSearchRequest,
    db: AsyncSession = Depends(get_db),
) -> HybridSearchResponse:
    """Combines vector similarity and keyword search using Reciprocal Rank Fusion."""
    engine = HybridSearchEngine(db)
    return await engine.search(request)


@router.post(
    "/query",
    response_model=EngineeringAnswerResponse,
    status_code=status.HTTP_200_OK,
    summary="Ask an evidence-grounded question about engineering history",
)
async def answer_engineering_question(
    request: EngineeringQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> EngineeringAnswerResponse:
    """Answer natural language engineering questions backed by retrieved PR evidence and AI understandings."""
    service = EngineeringRAGService(db)
    return await service.answer_engineering_query(request)


@router.post(
    "/query/stream",
    summary="Stream an evidence-grounded answer to an engineering question with SSE",
)
async def stream_engineering_question(
    request: EngineeringQueryRequest,
    db: AsyncSession = Depends(get_db),
) -> StreamingResponse:
    """Stream real-time tokens and evidence metadata for engineering intelligence queries."""
    service = EngineeringRAGService(db)
    return StreamingResponse(
        service.stream_engineering_query(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )



