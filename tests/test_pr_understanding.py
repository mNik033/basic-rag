from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock, MagicMock
import pytest
import pytest_asyncio
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository
from app.domain.understanding import EvidenceType, PRUnderstandingProcessRequest, PRUnderstandingResult
from app.services.github.understanding_service import PRUnderstandingService


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


@pytest.mark.asyncio
async def test_pr_understanding_extraction_and_save(test_db_session: AsyncSession):
    # Setup sample Repo and PR in DB
    repo = Repository(owner="meta", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=1234,
        number=42,
        title="Optimize Fiber reconcile memory footprint",
        body="This reduces memory overhead during large list renders by reusing existing fiber nodes.",
        state="closed",
        author="dan_abramov",
        merged_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        labels=["performance", "core"],
    )
    test_db_session.add(pr)
    await test_db_session.flush()

    # Add commit and changed file
    test_db_session.add(
        Commit(
            pull_request_id=pr.id,
            sha="1234567890abcdef",
            message="perf(reconciler): recycle fiber instances",
            additions=30,
            deletions=10,
        )
    )
    test_db_session.add(
        ChangedFile(
            pull_request_id=pr.id,
            filename="packages/react-reconciler/src/ReactFiber.js",
            status="modified",
            additions=30,
            deletions=10,
            changes=40,
            patch_text="@@ -10,2 +10,12 @@ function createFiber() {",
        )
    )
    await test_db_session.commit()

    # Mock LLM Response
    mock_llm_payload = {
        "summary": "Optimized memory footprint during reconciler updates by recycling fiber instances.",
        "motivation": {
            "evidence_type": "documented",
            "reason": "Reduces memory overhead during large list renders by reusing existing fiber nodes",
            "evidence_quote": "This reduces memory overhead during large list renders",
        },
        "components": ["ReactFiber", "ReactReconciler"],
        "change_types": ["memory", "performance"],
        "impact": ["lower memory churn during reconciler tree updates"],
        "architectural_change": False,
        "breaking_change": False,
        "key_technical_details": ["fiber node recycling pool", "decreased allocation frequency"],
    }

    mock_llm = MagicMock()
    mock_llm.model = "gemma4:e2b"
    mock_llm.generate_response = AsyncMock(return_value=json.dumps(mock_llm_payload))

    service = PRUnderstandingService(test_db_session, llm_service=mock_llm)

    # Test single analysis
    result = await service.analyze_pull_request(pr)
    assert result.summary == mock_llm_payload["summary"]
    assert result.motivation.evidence_type == EvidenceType.DOCUMENTED
    assert "memory" in result.change_types
    assert "ReactFiber" in result.components

    # Test batch execution & saving to DB
    response = await service.process_batch(
        PRUnderstandingProcessRequest(owner="meta", repo="react")
    )
    assert response.processed_count == 1
    assert response.status == "completed"

    # Verify saved row in DB
    stmt = select(PRUnderstanding).where(PRUnderstanding.pull_request_id == pr.id)
    saved_row = (await test_db_session.execute(stmt)).scalar_one()

    assert saved_row.motivation_type == "documented"
    assert saved_row.motivation_quote == "This reduces memory overhead during large list renders"


@pytest.mark.asyncio
async def test_smart_diff_filtering(test_db_session: AsyncSession):
    repo = Repository(owner="test", name="diff-filter")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=999,
        number=1,
        title="Update deps and app core",
        body="Upgraded dependencies and modified engine.",
        state="closed",
        author="alice",
        created_at=datetime.now(timezone.utc),
    )
    test_db_session.add(pr)
    await test_db_session.flush()

    # Add lockfile (should be skipped), minified JS (skipped), and python file (included)
    test_db_session.add(
        ChangedFile(
            pull_request_id=pr.id,
            filename="package-lock.json",
            status="modified",
            additions=500,
            deletions=200,
            changes=700,
            patch_text="@@ lockfile noise @@",
        )
    )
    test_db_session.add(
        ChangedFile(
            pull_request_id=pr.id,
            filename="dist/bundle.min.js",
            status="modified",
            additions=100,
            deletions=100,
            changes=200,
            patch_text="@@ minified noise @@",
        )
    )
    test_db_session.add(
        ChangedFile(
            pull_request_id=pr.id,
            filename="app/core/engine.py",
            status="modified",
            additions=15,
            deletions=3,
            changes=18,
            patch_text="@@ -1,3 +1,15 @@ def run_engine():",
        )
    )
    await test_db_session.commit()

    service = PRUnderstandingService(test_db_session)
    prompt = service.build_pr_context_prompt(pr)

    # Verify python source is in prompt diff
    assert "app/core/engine.py" in prompt
    assert "def run_engine():" in prompt

    # Verify lockfile and minified diff patches are omitted
    assert "package-lock.json (modified" not in prompt
    assert "dist/bundle.min.js (modified" not in prompt
    assert "non-source / generated file(s) omitted from diff" in prompt

    assert "performance" in saved_row.change_types
    assert saved_row.model_used == "gemma4:e2b"
