from datetime import datetime, timezone
import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.database import Base
from app.domain.github import (
    GitHubSyncRequest,
    RawChangedFile,
    RawCommit,
    RawPullRequest,
    RawReview,
    RawReviewComment,
)
from app.domain.models import ChangedFile, Commit, PullRequest, Repository, Review, ReviewComment
from app.services.github.collector import GitHubCollectorService


@pytest.fixture
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
async def test_collector_upsert_pr(test_db_session: AsyncSession):
    collector = GitHubCollectorService(test_db_session)
    repo = await collector.get_or_create_repository("test-owner", "test-repo")

    raw_pr = RawPullRequest(
        github_pr_id=101,
        number=1,
        title="Fix memory leak in ImageCache",
        body="Resolves buffer overflow issue.",
        state="closed",
        author="alice",
        merged_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
        commits=[
            RawCommit(
                sha="abc1234",
                author_name="alice",
                message="fix(cache): free allocated pointers",
                additions=10,
                deletions=2,
            )
        ],
        changed_files=[
            RawChangedFile(
                filename="src/cache.cpp",
                status="modified",
                additions=10,
                deletions=2,
                changes=12,
                patch_text="@@ -1,2 +1,10 @@",
            )
        ],
        reviews=[
            RawReview(
                github_review_id=555,
                author="bob",
                state="APPROVED",
                body="LGTM!",
                submitted_at=datetime.now(timezone.utc),
            )
        ],
        review_comments=[
            RawReviewComment(
                github_comment_id=999,
                review_id=555,
                author="bob",
                path="src/cache.cpp",
                line=5,
                body="Nice optimization.",
                created_at=datetime.now(timezone.utc),
            )
        ],
    )

    # 1. Test insertion
    saved = await collector.upsert_pull_request(repo.id, raw_pr)
    assert saved is True
    await test_db_session.commit()

    # Query back
    stmt = select(PullRequest).where(PullRequest.repository_id == repo.id, PullRequest.number == 1)
    res = await test_db_session.execute(stmt)
    pr = res.scalar_one()

    assert pr.title == "Fix memory leak in ImageCache"
    assert pr.author == "alice"

    # Query commits
    c_stmt = select(Commit).where(Commit.pull_request_id == pr.id)
    commits = (await test_db_session.execute(c_stmt)).scalars().all()
    assert len(commits) == 1
    assert commits[0].sha == "abc1234"

    # Query changed files
    f_stmt = select(ChangedFile).where(ChangedFile.pull_request_id == pr.id)
    files = (await test_db_session.execute(f_stmt)).scalars().all()
    assert len(files) == 1
    assert files[0].filename == "src/cache.cpp"
    assert files[0].patch_text == "@@ -1,2 +1,10 @@"

    # Query reviews & comments
    r_stmt = select(Review).where(Review.pull_request_id == pr.id)
    reviews = (await test_db_session.execute(r_stmt)).scalars().all()
    assert len(reviews) == 1
    assert reviews[0].author == "bob"

    rc_stmt = select(ReviewComment).where(ReviewComment.pull_request_id == pr.id)
    comments = (await test_db_session.execute(rc_stmt)).scalars().all()
    assert len(comments) == 1
    assert comments[0].body == "Nice optimization."


@pytest.mark.asyncio
async def test_collector_idempotency(test_db_session: AsyncSession):
    collector = GitHubCollectorService(test_db_session)
    repo = await collector.get_or_create_repository("test-owner", "test-repo")

    raw_pr = RawPullRequest(
        github_pr_id=202,
        number=2,
        title="Add metric logging",
        state="closed",
        author="charlie",
        merged_at=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )

    # First insert
    saved1 = await collector.upsert_pull_request(repo.id, raw_pr, force_resync=False)
    assert saved1 is True

    # Duplicate run without force_resync -> should skip
    saved2 = await collector.upsert_pull_request(repo.id, raw_pr, force_resync=False)
    assert saved2 is False
