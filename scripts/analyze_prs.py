import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Ensure workspace root is on sys.path when executed directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import async_session_factory, init_db
from app.domain.understanding import PRUnderstandingProcessRequest
from app.services.github.understanding_service import PRUnderstandingService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("analyze_prs")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run LLM PR Understanding over ingested GitHub pull requests."
    )
    parser.add_argument(
        "--owner",
        type=str,
        required=True,
        help="Repository owner / organization",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="Repository name",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="Specific PR number to analyze (default: all unanalyzed)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Max number of PRs to analyze in this batch (default: 50)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-analyzing PRs that were already summarized",
    )

    args = parser.parse_args()

    await init_db()

    request = PRUnderstandingProcessRequest(
        owner=args.owner,
        repo=args.repo,
        pr_number=args.pr,
        limit=args.limit,
        force_reprocess=args.force,
    )

    logger.info("Starting PR Understanding batch for %s/%s...", args.owner, args.repo)

    async with async_session_factory() as session:
        service = PRUnderstandingService(session)
        response = await service.process_batch(request)

    logger.info("Processing completed! Result: %s", response.message)


if __name__ == "__main__":
    asyncio.run(main())
