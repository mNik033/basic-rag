import asyncio
import logging
from datetime import datetime, timezone
import time
from typing import Any, AsyncGenerator, Dict, List, Optional
import httpx

from app.core.config import get_settings
from app.domain.github import (
    RawChangedFile,
    RawCommit,
    RawPullRequest,
    RawReview,
    RawReviewComment,
)

logger = logging.getLogger(__name__)


class GitHubRateLimitError(Exception):
    """Raised when GitHub API rate limits are encountered."""
    pass


class GitHubApiClient:
    """Asynchronous client for querying GitHub REST API v3 with rate-limit and retry handling."""

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: Optional[int] = None,
        pause_seconds: Optional[int] = None,
    ) -> None:
        settings = get_settings()
        self.token = token or settings.github_token
        self.base_url = (base_url or settings.github_api_base_url).rstrip("/")
        self.max_retries = max_retries if max_retries is not None else settings.github_max_retries
        self.pause_seconds = pause_seconds if pause_seconds is not None else settings.github_rate_limit_pause_seconds
        self.per_page = settings.github_per_page

        headers = {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "BasicRAG-Engineering-Memory",
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers=headers,
            timeout=httpx.Timeout(30.0, connect=10.0),
        )

    async def close(self) -> None:
        await self.client.aclose()

    async def __aenter__(self) -> "GitHubApiClient":
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        await self.close()

    async def _handle_rate_limits(self, response: httpx.Response) -> None:
        """Inspect rate limit headers and wait if rate limit has been reached."""
        remaining = response.headers.get("x-ratelimit-remaining")
        reset_ts = response.headers.get("x-ratelimit-reset")

        if remaining is not None and int(remaining) == 0 and reset_ts is not None:
            now_ts = int(time.time())
            sleep_duration = max(int(reset_ts) - now_ts, 1)
            logger.warning(
                "GitHub API rate limit reached (0 remaining). Sleeping for %s seconds until reset.",
                sleep_duration,
            )
            await asyncio.sleep(min(sleep_duration, self.pause_seconds))

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[Dict[str, Any]] = None,
    ) -> httpx.Response:
        """Execute an HTTP request with exponential backoff and rate-limit handling."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        last_exception = None

        for attempt in range(1, self.max_retries + 1):
            try:
                response = await self.client.request(method, url, params=params)
                await self._handle_rate_limits(response)

                if response.status_code == 403 and "rate limit exceeded" in response.text.lower():
                    logger.warning("Rate limit exceeded on attempt %s/%s. Backing off.", attempt, self.max_retries)
                    await asyncio.sleep(self.pause_seconds * attempt)
                    continue

                if response.status_code == 429:
                    retry_after = int(response.headers.get("Retry-After", self.pause_seconds))
                    logger.warning("Secondary rate limit (429) hit. Waiting %s seconds.", retry_after)
                    await asyncio.sleep(retry_after)
                    continue

                response.raise_for_status()
                return response

            except (httpx.RequestError, httpx.HTTPStatusError) as exc:
                last_exception = exc
                status_code = getattr(getattr(exc, "response", None), "status_code", None)
                if status_code and status_code in (401, 404):
                    # Do not retry client auth/not-found errors
                    raise

                wait_time = 2 ** attempt
                logger.warning(
                    "Request %s %s failed (attempt %s/%s): %s. Retrying in %ss.",
                    method, url, attempt, self.max_retries, exc, wait_time
                )
                await asyncio.sleep(wait_time)

        raise GitHubRateLimitError(f"GitHub API request failed after {self.max_retries} attempts: {last_exception}")

    async def get_repository_details(self, owner: str, repo: str) -> Dict[str, Any]:
        """Fetch general repository details."""
        resp = await self._request("GET", f"/repos/{owner}/{repo}")
        return resp.json()

    async def paginate(
        self,
        path: str,
        params: Optional[Dict[str, Any]] = None,
        max_items: Optional[int] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Iterate over GitHub API pages yielding individual items."""
        query_params = dict(params or {})
        query_params.setdefault("per_page", self.per_page)
        query_params.setdefault("page", 1)

        total_yielded = 0

        while True:
            resp = await self._request("GET", path, params=query_params)
            items = resp.json()

            if not isinstance(items, list) or len(items) == 0:
                break

            for item in items:
                yield item
                total_yielded += 1
                if max_items and total_yielded >= max_items:
                    return

            # Check Link header for rel="next"
            link_header = resp.headers.get("link", "")
            if 'rel="next"' not in link_header:
                break

            query_params["page"] += 1

    async def fetch_pull_requests(
        self,
        owner: str,
        repo: str,
        state: str = "closed",
        sort: str = "created",
        direction: str = "desc",
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Fetch pull requests matching state and ordering."""
        params = {
            "state": state,
            "sort": sort,
            "direction": direction,
        }
        async for pr in self.paginate(f"/repos/{owner}/{repo}/pulls", params=params):
            yield pr

    async def fetch_pr_commits(self, owner: str, repo: str, pr_number: int) -> List[RawCommit]:
        """Fetch all commits for a given pull request."""
        commits: List[RawCommit] = []
        async for c in self.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/commits"):
            commit_data = c.get("commit", {})
            author_data = commit_data.get("author", {})
            committed_at = None
            if author_data.get("date"):
                try:
                    committed_at = datetime.fromisoformat(author_data["date"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            stats = c.get("stats", {})
            commits.append(
                RawCommit(
                    sha=c["sha"],
                    author_name=author_data.get("name") or (c.get("author") or {}).get("login"),
                    author_email=author_data.get("email"),
                    message=commit_data.get("message", ""),
                    committed_at=committed_at,
                    additions=stats.get("additions", 0),
                    deletions=stats.get("deletions", 0),
                )
            )
        return commits

    async def fetch_pr_files(self, owner: str, repo: str, pr_number: int) -> List[RawChangedFile]:
        """Fetch all changed files and diff patches for a pull request."""
        files: List[RawChangedFile] = []
        async for f in self.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/files"):
            files.append(
                RawChangedFile(
                    filename=f["filename"],
                    status=f.get("status", "modified"),
                    additions=f.get("additions", 0),
                    deletions=f.get("deletions", 0),
                    changes=f.get("changes", 0),
                    patch_text=f.get("patch"),
                )
            )
        return files

    async def fetch_pr_reviews(self, owner: str, repo: str, pr_number: int) -> List[RawReview]:
        """Fetch code reviews for a pull request."""
        reviews: List[RawReview] = []
        async for r in self.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/reviews"):
            submitted_at = None
            if r.get("submitted_at"):
                try:
                    submitted_at = datetime.fromisoformat(r["submitted_at"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            reviews.append(
                RawReview(
                    github_review_id=r["id"],
                    author=(r.get("user") or {}).get("login", "unknown"),
                    state=r.get("state", "COMMENTED"),
                    body=r.get("body"),
                    submitted_at=submitted_at,
                )
            )
        return reviews

    async def fetch_pr_review_comments(
        self, owner: str, repo: str, pr_number: int
    ) -> List[RawReviewComment]:
        """Fetch inline review comments for a pull request."""
        comments: List[RawReviewComment] = []
        async for c in self.paginate(f"/repos/{owner}/{repo}/pulls/{pr_number}/comments"):
            created_at = datetime.now(timezone.utc)
            if c.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(c["created_at"].replace("Z", "+00:00"))
                except ValueError:
                    pass

            comments.append(
                RawReviewComment(
                    github_comment_id=c["id"],
                    review_id=c.get("pull_request_review_id"),
                    author=(c.get("user") or {}).get("login", "unknown"),
                    path=c.get("path"),
                    line=c.get("line") or c.get("original_line"),
                    body=c.get("body", ""),
                    created_at=created_at,
                )
            )
        return comments

    async def fetch_full_pr_details(
        self, owner: str, repo: str, pr_item: Dict[str, Any]
    ) -> RawPullRequest:
        """Fetch all child resources (commits, files, reviews, review comments) for a PR."""
        pr_number = pr_item["number"]

        # Fetch child collections
        commits_task = self.fetch_pr_commits(owner, repo, pr_number)
        files_task = self.fetch_pr_files(owner, repo, pr_number)
        reviews_task = self.fetch_pr_reviews(owner, repo, pr_number)
        comments_task = self.fetch_pr_review_comments(owner, repo, pr_number)

        commits, files, reviews, comments = await asyncio.gather(
            commits_task,
            files_task,
            reviews_task,
            comments_task,
        )

        def parse_iso(dt_str: Optional[str]) -> Optional[datetime]:
            if not dt_str:
                return None
            try:
                return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
            except ValueError:
                return None

        labels = [lbl["name"] for lbl in pr_item.get("labels", []) if isinstance(lbl, dict) and "name" in lbl]
        milestone = (pr_item.get("milestone") or {}).get("title") if pr_item.get("milestone") else None

        return RawPullRequest(
            github_pr_id=pr_item["id"],
            number=pr_number,
            title=pr_item.get("title", ""),
            body=pr_item.get("body"),
            state=pr_item.get("state", "closed"),
            author=(pr_item.get("user") or {}).get("login", "unknown"),
            merged_at=parse_iso(pr_item.get("merged_at")),
            merge_commit_sha=pr_item.get("merge_commit_sha"),
            labels=labels,
            milestone=milestone,
            created_at=parse_iso(pr_item.get("created_at")) or datetime.now(timezone.utc),
            closed_at=parse_iso(pr_item.get("closed_at")),
            commits=commits,
            changed_files=files,
            reviews=reviews,
            review_comments=comments,
            raw_json=pr_item,
        )
