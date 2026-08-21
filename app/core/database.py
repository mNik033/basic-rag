import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import DeclarativeBase

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class Base(DeclarativeBase):
    """Base declarative class for all SQLAlchemy ORM models."""
    pass


settings = get_settings()

engine = create_async_engine(
    settings.async_database_url,
    echo=settings.db_echo,
    future=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autocommit=False,
    autoflush=False,
)


async def _ensure_postgres_database_exists() -> None:
    """Ensure the target PostgreSQL database exists; create it if missing."""
    if "postgresql" not in settings.async_database_url:
        return

    target_db = settings.postgres_db
    # Maintenance connection to the default 'postgres' database
    maintenance_url = f"postgresql+asyncpg://{settings.postgres_user}:{settings.postgres_password}@{settings.postgres_host}:{settings.postgres_port}/postgres"

    try:
        maintenance_engine = create_async_engine(
            maintenance_url,
            isolation_level="AUTOCOMMIT",
        )
        async with maintenance_engine.connect() as conn:
            check_stmt = text("SELECT 1 FROM pg_database WHERE datname = :db_name")
            result = await conn.execute(check_stmt, {"db_name": target_db})
            exists = result.scalar() is not None

            if not exists:
                logger.info("Database '%s' does not exist. Creating database...", target_db)
                await conn.execute(text(f'CREATE DATABASE "{target_db}"'))
                logger.info("Database '%s' created successfully.", target_db)

        await maintenance_engine.dispose()
    except Exception as e:
        logger.warning(
            "Could not verify/create PostgreSQL database automatically: %s. "
            "Ensure the database exists or create it manually via: CREATE DATABASE %s;",
            e,
            target_db,
        )


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency yielding an async database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db() -> None:
    """Create all tables defined on Base (used for local setup and testing)."""
    await _ensure_postgres_database_exists()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

