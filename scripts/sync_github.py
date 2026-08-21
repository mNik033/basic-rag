import argparse
import asyncio
from datetime import datetime
import logging
from pathlib import Path
import sys

# Ensure workspace root is on sys.path when executed directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import async_session_factory, init_db
from app.domain.github import GitHubSyncRequest
from app.services.github.collector import GitHubCollectorService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sync_github")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest historical GitHub Pull Requests and engineering data into PostgreSQL."
    )
    parser.add_argument(
        "--owner",
        type=str,
        required=True,
        help="GitHub repository owner or organization (e.g. facebook)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository name (e.g. react)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="GitHub Personal Access Token (or set GITHUB_TOKEN in .env)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of merged PRs to fetch",
    )
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="ISO timestamp to filter merged PRs (e.g. 2024-01-01T00:00:00)",
    )
    parser.add_argument(
        "--force-resync",
        action="store_true",
        help="Force re-syncing of previously ingested PRs",
    )

    args = parser.parse_args()

    since_dt = None
    if args.since:
        try:
            since_dt = datetime.fromisoformat(args.since)
        except ValueError:
            logger.error("Invalid ISO timestamp for --since: %s", args.since)
            sys.exit(1)

    logger.info("Initializing database schema if not present...")
    await init_db()

    sync_request = GitHubSyncRequest(
        owner=args.owner,
        repo=args.repo,
        token=args.token,
        limit=args.limit,
        since=since_dt,
        force_resync=args.force_resync,
    )

    logger.info("Starting synchronization for %s/%s...", args.owner, args.repo)

    async with async_session_factory() as session:
        collector = GitHubCollectorService(session)
        result = await collector.sync_repository(sync_request)

    logger.info(
        "Sync completed successfully! Processed: %d PRs, %d Commits, %d Files, %d Reviews.",
        result.prs_processed,
        result.commits_processed,
        result.files_processed,
        result.reviews_processed,
    )


if __name__ == "__main__":
    asyncio.run(main())
