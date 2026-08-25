from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository
from app.domain.retrieval import (
    HybridSearchRequest,
    RetrievalFilter,
)
from app.services.retrieval.hybrid_search import HybridSearchEngine


@pytest_asyncio.fixture
async def test_db_session():
    """In-memory SQLite async session for testing."""
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session

    await engine.dispose()


def test_extract_search_tokens():
    engine = HybridSearchEngine(session=MagicMock())
    tokens = engine.extract_search_tokens("Which PRs changed ImageCache.cpp in release 5.3?")
    # 'Which', 'in', 'release' are stopwords or common words, 'ImageCache.cpp' and '5.3' and 'ImageCache' should be kept
    assert "ImageCache.cpp" in tokens
    assert "5.3" in tokens
    assert "which" not in [t.lower() for t in tokens]


@pytest.mark.asyncio
async def test_hybrid_search_vector_and_keyword_fusion(test_db_session: AsyncSession):
    # 1. Setup sample Repositories, PRs, ChangedFiles, and Understandings
    repo = Repository(owner="facebook", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    # PR #834: Changed ImageCache eviction
    pr1 = PullRequest(
        repository_id=repo.id,
        github_pr_id=8340,
        number=834,
        title="Optimize ImageCache eviction strategy",
        body="Reduces memory consumption on low-memory devices by tuning LRU cache eviction.",
        state="closed",
        author="alice",
        merged_at=datetime.now(timezone.utc),
        milestone="5.3",
    )
    test_db_session.add(pr1)
    await test_db_session.flush()

    file1 = ChangedFile(
        pull_request_id=pr1.id,
        filename="src/cache/ImageCache.cpp",
        status="modified",
        additions=45,
        deletions=12,
    )
    test_db_session.add(file1)

    und1 = PRUnderstanding(
        pull_request_id=pr1.id,
        summary="Optimized image cache eviction strategy for low-memory environments.",
        motivation_type="documented",
        motivation_reason="Avoid out-of-memory crashes on resource-constrained hardware.",
        components=["ImageCache", "MemoryManager"],
        change_types=["performance", "memory"],
        impact=["Reduced heap allocation", "Faster cache hits"],
        architectural_change=True,
        breaking_change=False,
        key_technical_details=["Replaced standard map with LRU doubly-linked list"],
        model_used="gemma4:e2b",
    )
    test_db_session.add(und1)

    # PR #621: Related bitmap memory issue
    pr2 = PullRequest(
        repository_id=repo.id,
        github_pr_id=6210,
        number=621,
        title="Fix bitmap retention leak during navigation",
        body="Prevents stale bitmap instances from lingering after screen transitions.",
        state="closed",
        author="bob",
        merged_at=datetime.now(timezone.utc),
        milestone="5.2",
    )
    test_db_session.add(pr2)
    await test_db_session.flush()

    und2 = PRUnderstanding(
        pull_request_id=pr2.id,
        summary="Addressed bitmap memory retention bug during UI navigation.",
        motivation_type="inferred",
        motivation_reason="Prevent gradual memory buildup leading to OOM.",
        components=["BitmapDecoder", "Navigation"],
        change_types=["bugfix", "memory"],
        impact=["Fixed memory leak on route transitions"],
        architectural_change=False,
        breaking_change=False,
        key_technical_details=["Explicitly dereference bitmap pointer on unmount"],
        model_used="gemma4:e2b",
    )
    test_db_session.add(und2)
    await test_db_session.commit()

    # 2. Mock embedding service and vector store
    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2, 0.3]

    mock_vector_store = MagicMock()
    # Vector search returns PR #834 as top hit, PR #621 as second hit
    mock_vector_store.query.return_value = {
        "ids": [["gh_pr_facebook_react_834", "gh_pr_facebook_react_621"]],
        "documents": [["# Repository: facebook/react | PR #834...", "# Repository: facebook/react | PR #621..."]],
        "metadatas": [[
            {
                "repository": "facebook/react",
                "pr_number": 834,
                "pr_title": "Optimize ImageCache eviction strategy",
                "author": "alice",
                "state": "closed",
                "components": "ImageCache,MemoryManager",
                "change_types": "performance,memory",
                "architectural_change": True,
            },
            {
                "repository": "facebook/react",
                "pr_number": 621,
                "pr_title": "Fix bitmap retention leak during navigation",
                "author": "bob",
                "state": "closed",
                "components": "BitmapDecoder,Navigation",
                "change_types": "bugfix,memory",
                "architectural_change": False,
            }
        ]],
        "distances": [[0.15, 0.35]],
    }

    engine = HybridSearchEngine(
        session=test_db_session,
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
    )

    # 3. Test Keyword Search independently
    kw_hits = await engine.search_keywords(query="ImageCache.cpp memory eviction")
    assert len(kw_hits) > 0
    assert kw_hits[0]["pr_number"] == 834
    assert any("File match" in aspect or "ImageCache" in aspect for aspect in kw_hits[0]["matched_aspects"])

    # 4. Test Hybrid Search Request (Vector + Keyword + RRF Fusion)
    request = HybridSearchRequest(
        query="Which PRs optimized memory for ImageCache?",
        limit=5,
        vector_weight=0.6,
        keyword_weight=0.4,
    )
    response = await engine.search(request)

    assert response.total_candidates >= 2
    assert response.results[0].pr_number == 834
    assert response.results[0].repository == "facebook/react"
    assert response.results[0].rank == 1
    assert response.results[0].combined_score == 1.0  # normalized top score
    assert "ImageCache" in response.results[0].components
    assert response.results[0].architectural_change is True
    assert any("Dual-modality" in reason for reason in response.results[0].match_reasons)

    # Verify second candidate
    assert response.results[1].pr_number == 621
    assert response.results[1].rank == 2


@pytest.mark.asyncio
async def test_hybrid_search_with_metadata_filters(test_db_session: AsyncSession):
    repo = Repository(owner="meta", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=900,
        number=900,
        title="Refactor core scheduler",
        body="Major architecture rewrite of priority queue scheduler.",
        state="closed",
        author="sophiebits",
        merged_at=datetime.now(timezone.utc),
    )
    test_db_session.add(pr)
    await test_db_session.flush()

    und = PRUnderstanding(
        pull_request_id=pr.id,
        summary="Rewrote scheduler priority queue.",
        motivation_type="documented",
        motivation_reason="Improve frame timing.",
        components=["Scheduler"],
        change_types=["refactor"],
        architectural_change=True,
        breaking_change=False,
    )
    test_db_session.add(und)
    await test_db_session.commit()

    mock_embedding = MagicMock()
    mock_embedding.embed_text.return_value = [0.1, 0.2]
    mock_vector_store = MagicMock()
    mock_vector_store.query.return_value = {
        "ids": [["gh_pr_meta_react_900"]],
        "documents": [["# PR #900"]],
        "metadatas": [[{
            "repository": "meta/react",
            "pr_number": 900,
            "pr_title": "Refactor core scheduler",
            "author": "sophiebits",
            "state": "closed",
        }]],
        "distances": [[0.1]],
    }

    engine = HybridSearchEngine(
        session=test_db_session,
        embedding_service=mock_embedding,
        vector_store=mock_vector_store,
    )

    # Filter with matching architectural_only=True
    filter_match = RetrievalFilter(owner="meta", repo="react", architectural_only=True)
    res_match = await engine.search(HybridSearchRequest(query="scheduler", filter=filter_match))
    assert res_match.total_candidates == 1
    assert res_match.results[0].pr_number == 900

    # Filter with non-matching breaking_only=True
    filter_no_match = RetrievalFilter(breaking_only=True)
    res_no_match = await engine.search(HybridSearchRequest(query="scheduler", filter=filter_no_match))
    # Vector store query called with where={"breaking_change": True}
    assert mock_vector_store.query.called
