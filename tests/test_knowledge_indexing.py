from datetime import datetime, timezone
from unittest.mock import MagicMock
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base, get_db
from app.domain.knowledge import (
    EngineeringDocument,
    KnowledgeIndexRequest,
)
from app.domain.models import ChangedFile, Commit, PRUnderstanding, PullRequest, Repository
from app.main import app
from app.services.github.knowledge_builder import EngineeringDocumentSynthesizer
from app.services.github.knowledge_service import KnowledgeBaseService


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


def test_engineering_document_synthesizer():
    repo = Repository(owner="facebook", name="react")
    pr = PullRequest(
        repository_id=1,
        github_pr_id=101,
        number=42,
        title="Improve Fiber reconcile algorithm",
        body="This change reduces memory churn on reconciliation.",
        state="closed",
        author="sophiebits",
        merged_at=datetime.now(timezone.utc),
        milestone="v18.0",
        labels=["performance", "core"],
    )
    pr.commits = [
        Commit(sha="abc123456789", message="optimize reconcile memory", additions=20, deletions=5)
    ]
    pr.changed_files = [
        ChangedFile(filename="packages/react-reconciler/src/ReactFiber.js", status="modified", additions=20, deletions=5)
    ]
    understanding = PRUnderstanding(
        summary="Optimized React Fiber node allocation to reduce memory pressure.",
        motivation_type="documented",
        motivation_reason="PR description specifies memory churn reduction.",
        motivation_quote="reduces memory churn on reconciliation",
        components=["FiberReconciler", "Memory"],
        change_types=["performance", "memory"],
        impact=["Reduced GC pauses", "Faster reconciliation"],
        architectural_change=True,
        breaking_change=False,
        key_technical_details=["Object pool recycling"],
    )

    synthesizer = EngineeringDocumentSynthesizer()
    doc = synthesizer.synthesize_pr_document(repo=repo, pr=pr, understanding=understanding)

    assert isinstance(doc, EngineeringDocument)
    assert doc.doc_id == "gh_pr_facebook_react_42"
    assert "PR #42: Improve Fiber reconcile algorithm" in doc.text
    assert "Engineering Summary:" in doc.text
    assert "Optimized React Fiber node allocation" in doc.text
    assert "Impacted Components: FiberReconciler, Memory" in doc.text
    assert "Change Categories: performance, memory" in doc.text
    assert "Architectural Change: Yes" in doc.text
    assert "Breaking Change: No" in doc.text
    assert "ReactFiber.js" in doc.text

    # Verify metadata
    assert doc.metadata["repository"] == "facebook/react"
    assert doc.metadata["owner"] == "facebook"
    assert doc.metadata["repo"] == "react"
    assert doc.metadata["pr_number"] == 42
    assert doc.metadata["architectural_change"] is True
    assert doc.metadata["breaking_change"] is False
    assert doc.metadata["has_ai_understanding"] is True


@pytest.mark.asyncio
async def test_knowledge_base_service_indexing_and_status(test_db_session: AsyncSession):
    # 1. Setup sample repo, PR, and understanding
    repo = Repository(owner="facebook", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=101,
        number=42,
        title="Improve Fiber reconcile algorithm",
        body="Reduces memory churn.",
        state="closed",
        author="sophiebits",
        merged_at=datetime.now(timezone.utc),
        labels=["performance"],
    )
    test_db_session.add(pr)
    await test_db_session.flush()

    understanding = PRUnderstanding(
        pull_request_id=pr.id,
        summary="Optimized React Fiber node allocation.",
        motivation_type="inferred",
        motivation_reason="Code refactor for memory efficiency.",
        components=["FiberReconciler"],
        change_types=["performance"],
        impact=["Lower memory overhead"],
        architectural_change=False,
        breaking_change=False,
        key_technical_details=["Reused fiber nodes"],
        model_used="gemma:7b",
    )
    test_db_session.add(understanding)
    await test_db_session.commit()

    # 2. Mock embedding service and vector store
    mock_embedding_service = MagicMock()
    mock_embedding_service.embed_documents.return_value = [[0.1, 0.2, 0.3]]
    mock_embedding_service.model_name = "test-embed-model"

    mock_vector_store = MagicMock()
    mock_vector_store.collection.get.return_value = {"ids": []}

    service = KnowledgeBaseService(
        session=test_db_session,
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
    )

    # 3. Test get_knowledge_status before indexing
    status_before = await service.get_knowledge_status(owner="facebook", repo="react")
    assert status_before.repository == "facebook/react"
    assert status_before.total_prs == 1
    assert status_before.understood_prs == 1
    assert status_before.indexed_vectors == 0

    # 4. Trigger indexing
    request = KnowledgeIndexRequest(owner="facebook", repo="react", limit=10)
    index_res = await service.index_repository_knowledge(request)

    assert index_res.status == "completed"
    assert index_res.documents_indexed == 1
    assert index_res.chunks_created == 1
    assert mock_vector_store.upsert_chunks.called

    # 5. Indexing again with force_reindex=False when already indexed
    mock_vector_store.collection.get.return_value = {"ids": ["gh_pr_facebook_react_42"]}
    index_res_skip = await service.index_repository_knowledge(request)
    assert index_res_skip.documents_indexed == 0
    assert "already indexed" in index_res_skip.message

    # 6. Indexing again with force_reindex=True
    request_force = KnowledgeIndexRequest(owner="facebook", repo="react", force_reindex=True)
    index_res_force = await service.index_repository_knowledge(request_force)
    assert index_res_force.documents_indexed == 1


@pytest.mark.asyncio
async def test_knowledge_api_endpoints(test_db_session: AsyncSession):
    # Setup test DB repository and PR
    repo = Repository(owner="meta", name="react")
    test_db_session.add(repo)
    await test_db_session.flush()

    pr = PullRequest(
        repository_id=repo.id,
        github_pr_id=202,
        number=10,
        title="Add hooks support",
        body="Initial implementation of React Hooks.",
        state="closed",
        author="sebmarkbage",
        merged_at=datetime.now(timezone.utc),
    )
    test_db_session.add(pr)
    await test_db_session.commit()

    async def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        # 1. Test status endpoint
        res = await ac.get("/api/v1/github/knowledge/status/meta/react")
        assert res.status_code == 200
        data = res.json()
        assert data["repository"] == "meta/react"
        assert data["total_prs"] == 1

        # 2. Test status endpoint for non-existent repo (404)
        res_404 = await ac.get("/api/v1/github/knowledge/status/meta/nonexistent")
        assert res_404.status_code == 404

        # 3. Test index endpoint for non-existent repo (404)
        res_idx_404 = await ac.post(
            "/api/v1/github/knowledge/index",
            json={"owner": "meta", "repo": "nonexistent"},
        )
        assert res_idx_404.status_code == 404

    app.dependency_overrides.clear()
