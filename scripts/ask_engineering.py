import argparse
import asyncio
import logging
from pathlib import Path
import sys

# Ensure workspace root is on sys.path when executed directly as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import async_session_factory, init_db
from app.domain.retrieval import EngineeringQueryRequest, RetrievalFilter
from app.services.github.rag_service import EngineeringRAGService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("ask_engineering")


async def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the Engineering Intelligence knowledge base with grounded PR citations."
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        default=None,
        help="Natural language engineering question (e.g. 'Which PRs optimized memory in ImageCache?')",
    )
    parser.add_argument(
        "--owner",
        type=str,
        default=None,
        help="GitHub repository owner (e.g. facebook)",
    )
    parser.add_argument(
        "--repo",
        type=str,
        default=None,
        help="GitHub repository name (e.g. react)",
    )
    parser.add_argument(
        "--component",
        type=str,
        default=None,
        help="Filter by impacted component (e.g. ImageCache, Scheduler)",
    )
    parser.add_argument(
        "--change-type",
        type=str,
        default=None,
        help="Filter by change type (e.g. memory, performance, bugfix)",
    )
    parser.add_argument(
        "--architectural-only",
        action="store_true",
        help="Retrieve only architectural changes",
    )
    parser.add_argument(
        "--breaking-only",
        action="store_true",
        help="Retrieve only breaking changes",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Max number of candidate PRs to retrieve as context (default: 5)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream tokens in real time from LLM",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger("app.services.github.rag_service").setLevel(logging.DEBUG)
        logging.getLogger("app.services.retrieval.hybrid_search").setLevel(logging.DEBUG)

    query_text = args.query
    if not query_text:
        print("\n=== Engineering Intelligence Assistant ===")
        try:
            query_text = input("Ask a question about engineering evolution: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            return

    if not query_text:
        print("Error: Question cannot be empty.")
        return

    await init_db()

    repo_scope = f"{args.owner}/{args.repo}" if (args.owner and args.repo) else None
    retrieval_filter = RetrievalFilter(
        owner=args.owner,
        repo=args.repo,
        component=args.component,
        change_type=args.change_type,
        architectural_only=args.architectural_only,
        breaking_only=args.breaking_only,
    )

    request = EngineeringQueryRequest(
        query=query_text,
        repository=repo_scope,
        filter=retrieval_filter,
        limit=args.limit,
    )

    async with async_session_factory() as session:
        service = EngineeringRAGService(session)

        print("\n" + "=" * 60)
        print(f"QUESTION: {query_text}")
        if repo_scope:
            print(f"SCOPE:    {repo_scope}")
        print("=" * 60 + "\n")

        if args.stream:
            print("Answer:")
            async for sse_chunk in service.stream_engineering_query(request):
                lines = sse_chunk.strip().split("\n")
                event_type = None
                for line in lines:
                    if line.startswith("event:"):
                        event_type = line.replace("event:", "").strip()
                    elif line.startswith("data:") and event_type == "token":
                        import json
                        try:
                            payload = json.loads(line.replace("data:", "").strip())
                            print(payload.get("token", ""), end="", flush=True)
                        except Exception:
                            pass
            print("\n")
        else:
            response = await service.answer_engineering_query(request)
            print(f"SCENARIO DETECTED: {response.scenario_detected.upper()}")
            print(f"MODEL:             {response.model_used}")
            print(f"EVIDENCE FOUND:    {response.total_evidence_count} PRs\n")
            print("ANSWER:")
            print("-" * 60)
            print(response.answer)
            print("-" * 60)

            if response.evidence:
                print("\nCITATIONS & EVIDENCE:")
                for idx, ev in enumerate(response.evidence, 1):
                    mot_info = f"[{ev.motivation_type.upper()}] {ev.motivation_reason}" if ev.motivation_reason else "N/A"
                    comps = ", ".join(ev.components) if ev.components else "None"
                    print(f"\n{idx}. PR #{ev.pr_number} ({ev.repository}): {ev.title}")
                    print(f"   Author: {ev.author} | Milestone: {ev.milestone or 'N/A'}")
                    print(f"   Components: {comps}")
                    print(f"   Motivation: {mot_info}")
                    if ev.changed_files:
                        print(f"   Changed Files: {', '.join(ev.changed_files[:4])}")
                    if ev.match_reasons:
                        print(f"   Match Reason: {ev.match_reasons[0]}")
            print("\n" + "=" * 60 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
