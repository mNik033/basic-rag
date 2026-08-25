import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Ensure workspace root is on sys.path when executed directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import async_session_factory, init_db
from app.domain.knowledge import KnowledgeIndexRequest
from app.services.github.knowledge_service import KnowledgeBaseService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("index_knowledge")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Trigger vector indexing and embedding generation for GitHub PR knowledge into ChromaDB."
    )
    parser.add_argument(
        "--owner",
        type=str,
        required=True,
        help="GitHub repository owner / organization (e.g. facebook)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        required=True,
        help="GitHub repository name (e.g. react)",
    )
    parser.add_argument(
        "--pr",
        type=int,
        default=None,
        help="Specific PR number to index (default: all PRs)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Max number of PRs to index in this batch (default: 100)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-indexing and overwriting existing vector embeddings",
    )
    parser.add_argument(
        "--status-only",
        action="store_true",
        help="Check and display current indexing status without generating embeddings",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app.services.github.knowledge_service").setLevel(logging.DEBUG)

    await init_db()

    async with async_session_factory() as session:
        service = KnowledgeBaseService(session)

        if args.status_only:
            logger.info("Fetching knowledge indexing status for %s/%s...", args.owner, args.repo)
            try:
                status_res = await service.get_knowledge_status(owner=args.owner, repo=args.repo)
                print("\n" + "=" * 50)
                print(f" Knowledge Status: {status_res.repository}")
                print("=" * 50)
                print(f" Total Ingested PRs : {status_res.total_prs}")
                print(f" Understood PRs     : {status_res.understood_prs}")
                print(f" Indexed PR Vectors : {status_res.indexed_vectors}")
                print("=" * 50 + "\n")
            except Exception as e:
                logger.error("Failed to retrieve knowledge status: %s", e)
                sys.exit(1)
            return

        request = KnowledgeIndexRequest(
            owner=args.owner,
            repo=args.repo,
            pr_number=args.pr,
            limit=args.limit,
            force_reindex=args.force,
        )

        logger.info("Starting Knowledge Base indexing for %s/%s...", args.owner, args.repo)
        try:
            result = await service.index_repository_knowledge(request)
            print("\n" + "=" * 50)
            print(f" Indexing Result: {result.repository}")
            print("=" * 50)
            print(f" Status            : {result.status}")
            print(f" Documents Indexed : {result.documents_indexed}")
            print(f" Chunks Created    : {result.chunks_created}")
            print(f" Message           : {result.message}")
            print("=" * 50)

            # Query updated status
            updated_status = await service.get_knowledge_status(owner=args.owner, repo=args.repo)
            print(f" Current Total Vectors in DB: {updated_status.indexed_vectors}\n")

        except Exception as e:
            logger.error("Knowledge indexing failed: %s", e)
            sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
