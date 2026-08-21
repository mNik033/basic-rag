from datetime import datetime, timezone
import logging
from typing import Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.domain.github import GitHubSyncRequest, GitHubSyncResponse, RawPullRequest
from app.domain.models import (
    ChangedFile,
    Commit,
    PullRequest,
    Repository,
    Review,
    ReviewComment,
    SyncState,
)
from app.services.github.client import GitHubApiClient

logger = logging.getLogger(__name__)


class GitHubCollectorService:
    """Coordinates fetching and storing GitHub engineering data in PostgreSQL."""

    def __init__(self, session: AsyncSession, client: Optional[GitHubApiClient] = None) -> None:
        self.session = session
        self.client = client

    async def get_or_create_repository(self, owner: str, name: str) -> Repository:
        """Find or insert a repository record."""
        stmt = select(Repository).where(Repository.owner == owner, Repository.name == name)
        res = await self.session.execute(stmt)
        repo = res.scalar_one_or_none()

        if not repo:
            # Fetch default branch and description if client is available
            default_branch = "main"
            description = None
            if self.client:
                try:
                    repo_info = await self.client.get_repository_details(owner, name)
                    default_branch = repo_info.get("default_branch", "main")
                    description = repo_info.get("description")
                except Exception as e:
                    logger.warning("Could not fetch remote repo details for %s/%s: %s", owner, name, e)

            repo = Repository(
                owner=owner,
                name=name,
                default_branch=default_branch,
                description=description,
            )
            self.session.add(repo)
            await self.session.flush()

        return repo

    async def get_or_create_sync_state(self, repository_id: int) -> SyncState:
        """Find or initialize the sync state record for a repository."""
        stmt = select(SyncState).where(SyncState.repository_id == repository_id)
        res = await self.session.execute(stmt)
        state = res.scalar_one_or_none()

        if not state:
            state = SyncState(repository_id=repository_id, status="idle")
            self.session.add(state)
            await self.session.flush()

        return state

    async def upsert_pull_request(
        self, repository_id: int, raw_pr: RawPullRequest, force_resync: bool = False
    ) -> bool:
        """Idempotently insert or update a pull request and all its child entities.

        Returns True if a new/updated PR was written, False if skipped.
        """
        stmt = select(PullRequest).where(
            PullRequest.repository_id == repository_id,
            PullRequest.number == raw_pr.number,
        )
        res = await self.session.execute(stmt)
        existing_pr = res.scalar_one_or_none()

        if existing_pr and not force_resync:
            logger.debug("PR #%s already synced for repo_id %s; skipping.", raw_pr.number, repository_id)
            return False

        if existing_pr:
            # Update existing PR fields
            existing_pr.title = raw_pr.title
            existing_pr.body = raw_pr.body
            existing_pr.state = raw_pr.state
            existing_pr.author = raw_pr.author
            existing_pr.merged_at = raw_pr.merged_at
            existing_pr.merge_commit_sha = raw_pr.merge_commit_sha
            existing_pr.labels = raw_pr.labels
            existing_pr.milestone = raw_pr.milestone
            existing_pr.closed_at = raw_pr.closed_at
            existing_pr.raw_json = raw_pr.raw_json
            pr_record = existing_pr
            # Clear child entities to re-insert freshly parsed data
            await self.session.delete(existing_pr)
            await self.session.flush()

        # Insert fresh PR record
        pr_record = PullRequest(
            repository_id=repository_id,
            github_pr_id=raw_pr.github_pr_id,
            number=raw_pr.number,
            title=raw_pr.title,
            body=raw_pr.body,
            state=raw_pr.state,
            author=raw_pr.author,
            merged_at=raw_pr.merged_at,
            merge_commit_sha=raw_pr.merge_commit_sha,
            labels=raw_pr.labels,
            milestone=raw_pr.milestone,
            created_at=raw_pr.created_at,
            closed_at=raw_pr.closed_at,
            raw_json=raw_pr.raw_json,
        )
        self.session.add(pr_record)
        await self.session.flush()

        # Commits
        for c in raw_pr.commits:
            self.session.add(
                Commit(
                    pull_request_id=pr_record.id,
                    sha=c.sha,
                    author_name=c.author_name,
                    author_email=c.author_email,
                    message=c.message,
                    committed_at=c.committed_at,
                    additions=c.additions,
                    deletions=c.deletions,
                )
            )

        # Changed Files
        for f in raw_pr.changed_files:
            self.session.add(
                ChangedFile(
                    pull_request_id=pr_record.id,
                    filename=f.filename,
                    status=f.status,
                    additions=f.additions,
                    deletions=f.deletions,
                    changes=f.changes,
                    patch_text=f.patch_text,
                )
            )

        # Reviews & Map review ids
        review_id_map = {}
        for r in raw_pr.reviews:
            review_entity = Review(
                pull_request_id=pr_record.id,
                github_review_id=r.github_review_id,
                author=r.author,
                state=r.state,
                body=r.body,
                submitted_at=r.submitted_at,
            )
            self.session.add(review_entity)
            await self.session.flush()
            review_id_map[r.github_review_id] = review_entity.id

        # Review Comments
        for rc in raw_pr.review_comments:
            mapped_review_id = review_id_map.get(rc.review_id) if rc.review_id else None
            self.session.add(
                ReviewComment(
                    pull_request_id=pr_record.id,
                    review_id=mapped_review_id,
                    github_comment_id=rc.github_comment_id,
                    author=rc.author,
                    path=rc.path,
                    line=rc.line,
                    body=rc.body,
                    created_at=rc.created_at,
                )
            )

        await self.session.flush()
        return True

    async def sync_repository(self, request: GitHubSyncRequest) -> GitHubSyncResponse:
        """Run full or incremental sync for the specified repository."""
        repo = await self.get_or_create_repository(request.owner, request.repo)
        sync_state = await self.get_or_create_sync_state(repo.id)

        sync_state.status = "in_progress"
        sync_state.error_message = None
        await self.session.commit()

        client_to_use = self.client or GitHubApiClient(token=request.token)
        close_client = self.client is None

        prs_synced = 0
        total_commits = 0
        total_files = 0
        total_reviews = 0
        highest_pr_number = sync_state.last_synced_pr_number or 0

        try:
            async with client_to_use:
                async for pr_summary in client_to_use.fetch_pull_requests(
                    owner=request.owner,
                    repo=request.repo,
                    state="closed",
                    limit=request.limit,
                ):
                    # Only process merged PRs (merged_at is not null)
                    if not pr_summary.get("merged_at"):
                        continue

                    # Filter by timestamp if requested
                    merged_dt = None
                    if pr_summary.get("merged_at"):
                        try:
                            merged_dt = datetime.fromisoformat(
                                pr_summary["merged_at"].replace("Z", "+00:00")
                            )
                        except ValueError:
                            pass

                    if request.since and merged_dt and merged_dt < request.since:
                        break

                    pr_num = pr_summary["number"]
                    logger.info("Ingesting PR #%s (%s)", pr_num, pr_summary.get("title"))

                    full_pr = await client_to_use.fetch_full_pr_details(
                        request.owner, request.repo, pr_summary
                    )

                    was_saved = await self.upsert_pull_request(
                        repo.id, full_pr, force_resync=request.force_resync
                    )

                    if was_saved:
                        prs_synced += 1
                        total_commits += len(full_pr.commits)
                        total_files += len(full_pr.changed_files)
                        total_reviews += len(full_pr.reviews)
                        if pr_num > highest_pr_number:
                            highest_pr_number = pr_num

                    # Commit every batch of 10 PRs to save progress
                    if prs_synced % 10 == 0:
                        await self.session.commit()

            # Mark sync complete
            sync_state.status = "success"
            sync_state.last_synced_pr_number = highest_pr_number
            sync_state.total_prs_synced += prs_synced
            sync_state.last_synced_at = datetime.now(timezone.utc)
            await self.session.commit()

            return GitHubSyncResponse(
                repository=f"{request.owner}/{request.repo}",
                status="success",
                message=f"Successfully synced {prs_synced} merged pull requests.",
                total_synced=sync_state.total_prs_synced,
                prs_processed=prs_synced,
                commits_processed=total_commits,
                files_processed=total_files,
                reviews_processed=total_reviews,
            )

        except Exception as exc:
            logger.exception("GitHub sync failed for %s/%s: %s", request.owner, request.repo, exc)
            await self.session.rollback()
            sync_state.status = "failed"
            sync_state.error_message = str(exc)
            await self.session.commit()
            raise
        finally:
            if close_client:
                await client_to_use.close()
