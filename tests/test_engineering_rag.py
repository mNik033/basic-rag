import json
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository
from app.domain.retrieval import (
    EngineeringAnswerResponse,
    EngineeringEvidence,
    EngineeringQueryRequest,
    HybridSearchRequest,
    HybridSearchResponse,
    RetrievalCandidate,
    RetrievalFilter,
)
from app.main import app
from app.services.github.rag_service import (
    ENGINEERING_SYSTEM_PROMPT,
    EngineeringRAGService,
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


def test_scenario_classification():
    service = EngineeringRAGService(session=MagicMock())

    assert service.classify_scenario("What changed between release 5.2 and 5.3?") == "release_comparison"
    assert service.classify_scenario("Have we seen this issue before with memory leaks?") == "historical_issue"
    assert service.classify_scenario("Which PRs affected performance and memory in ImageCache?") == "impact_search"
    assert service.classify_scenario("Why was the scheduler architecture changed?") == "decision_understanding"
    assert service.classify_scenario("How does the codebase organize routing?") == "general_qa"


def test_build_context_prompt_formatting():
    service = EngineeringRAGService(session=MagicMock())

    candidate = RetrievalCandidate(
        pr_number=834,
        repository="facebook/react",
        title="Optimize ImageCache eviction strategy",
        author="alice",
        state="closed",
        merged_at=datetime(2025, 1, 15, 10, 0, 0, tzinfo=timezone.utc),
        milestone="5.3",
        summary="Optimized cache eviction to reduce memory consumption on low-memory devices.",
        motivation_type="documented",
        motivation_reason="Reduce memory consumption on mobile devices.",
        components=["ImageCache", "Gallery"],
        change_types=["memory", "performance"],
        architectural_change=True,
        breaking_change=False,
        key_technical_details=["Replaced std::map with LRU linked list"],
        changed_files=["src/ImageCache.cpp", "src/ImageCache.h"],
        combined_score=0.95,
        rank=1,
        match_reasons=["Keyword match: ImageCache"],
    )

    prompt = service.build_context_prompt(
        query="Have we seen image caching memory problems before?",
        candidates=[candidate],
        scenario="historical_issue",
    )

    assert "PR #834: Optimize ImageCache eviction strategy" in prompt
    assert "Motivation (Documented): Reduce memory consumption on mobile devices." in prompt
    assert "Impacted Components: ImageCache, Gallery" in prompt
    assert "Architectural Change: Yes" in prompt
    assert "src/ImageCache.cpp" in prompt


@pytest.mark.asyncio
async def test_answer_engineering_query_with_grounded_evidence(test_db_session: AsyncSession):
    # Setup mock hybrid search engine
    mock_hybrid_engine = MagicMock()
    mock_candidate = RetrievalCandidate(
        pr_number=834,
        repository="facebook/react",
        title="Optimize ImageCache eviction",
        author="alice",
        state="closed",
        merged_at=datetime.now(timezone.utc),
        summary="Tuned LRU cache eviction.",
        motivation_type="documented",
        motivation_reason="Prevent OOM on constrained devices.",
        components=["ImageCache"],
        change_types=["memory"],
        combined_score=1.0,
        rank=1,
        match_reasons=["Dual-modality match"],
    )
    mock_hybrid_engine.search = AsyncMock(return_value=HybridSearchResponse(
        query="memory issues in ImageCache",
        total_candidates=1,
        results=[mock_candidate],
        vector_hits=1,
        keyword_hits=1,
    ))

    # Setup mock LLM
    mock_llm = MagicMock()
    mock_llm.model = "claude-5-sonnet"
    mock_llm.generate_response = AsyncMock(
        return_value="Yes, PR #834 optimized the ImageCache eviction strategy to prevent OOM.\n\nEvidence:\n* PR #834"
    )

    service = EngineeringRAGService(
        session=test_db_session,
        hybrid_search_engine=mock_hybrid_engine,
        llm_service=mock_llm,
    )

    req = EngineeringQueryRequest(
        query="Have we seen memory issues in ImageCache?",
        repository="facebook/react",
    )
    response = await service.answer_engineering_query(req)

    assert response.query == req.query
    assert "PR #834" in response.answer
    assert response.has_sufficient_evidence is True
    assert response.total_evidence_count == 1
    assert response.evidence[0].pr_number == 834
    assert response.evidence[0].motivation_type == "documented"
    assert response.scenario_detected == "historical_issue"


@pytest.mark.asyncio
async def test_answer_engineering_query_empty_evidence(test_db_session: AsyncSession):
    mock_hybrid_engine = MagicMock()
    mock_hybrid_engine.search = AsyncMock(return_value=HybridSearchResponse(
        query="quantum computing optimization",
        total_candidates=0,
        results=[],
        vector_hits=0,
        keyword_hits=0,
    ))

    mock_llm = MagicMock()
    mock_llm.model = "gemma4:e2b"

    service = EngineeringRAGService(
        session=test_db_session,
        hybrid_search_engine=mock_hybrid_engine,
        llm_service=mock_llm,
    )

    req = EngineeringQueryRequest(query="quantum computing optimization")
    response = await service.answer_engineering_query(req)

    assert response.has_sufficient_evidence is False
    assert response.total_evidence_count == 0
    assert "no relevant Pull Requests" in response.answer


@pytest.mark.asyncio
async def test_stream_engineering_query(test_db_session: AsyncSession):
    mock_hybrid_engine = MagicMock()
    mock_candidate = RetrievalCandidate(
        pr_number=621,
        repository="meta/react",
        title="Fix bitmap retention leak",
        author="bob",
        state="closed",
        summary="Fixed bitmap retention.",
        components=["BitmapDecoder"],
        combined_score=0.9,
        rank=1,
    )
    mock_hybrid_engine.search = AsyncMock(return_value=HybridSearchResponse(
        query="bitmap retention",
        total_candidates=1,
        results=[mock_candidate],
    ))

    async def mock_token_generator(*args, **kwargs):
        tokens = ["PR #621 ", "fixed ", "the ", "bitmap ", "leak."]
        for t in tokens:
            yield t

    mock_llm = MagicMock()
    mock_llm.model = "gemma4:e2b"
    mock_llm.stream_response = mock_token_generator

    service = EngineeringRAGService(
        session=test_db_session,
        hybrid_search_engine=mock_hybrid_engine,
        llm_service=mock_llm,
    )

    req = EngineeringQueryRequest(query="Why was bitmap memory leaking?")
    events = []
    async for sse_chunk in service.stream_engineering_query(req):
        events.append(sse_chunk)

    assert any("event: metadata" in e for e in events)
    assert any("event: token" in e for e in events)
    assert any("event: done" in e for e in events)
    assert any("PR #621" in e for e in events)


@pytest.mark.asyncio
async def test_api_hybrid_search_and_query_endpoints(test_db_session: AsyncSession):
    # Seed test database
    repo = Repository(owner="facebook", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=8340,
        number=834,
        title="Optimize ImageCache eviction",
        body="Reduces memory consumption on low-memory devices.",
        state="closed",
        author="alice",
        merged_at=datetime.now(timezone.utc),
        milestone="5.3",
    )
    test_db_session.add(pr)
    await test_db_session.flush()

    und = PRUnderstanding(
        pull_request_id=pr.id,
        summary="Optimized cache eviction strategy.",
        motivation_type="documented",
        motivation_reason="Prevent OOM crashes.",
        components=["ImageCache"],
        change_types=["memory"],
    )
    test_db_session.add(und)
    await test_db_session.commit()

    async def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Test POST /api/v1/github/search/hybrid
        search_res = await ac.post(
            "/api/v1/github/search/hybrid",
            json={"query": "ImageCache memory eviction", "limit": 5},
        )
        assert search_res.status_code == 200
        search_data = search_res.json()
        assert search_data["query"] == "ImageCache memory eviction"
        assert search_data["total_candidates"] >= 1
        assert search_data["results"][0]["pr_number"] == 834

    app.dependency_overrides.clear()
