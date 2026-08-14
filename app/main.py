import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.exceptions import RAGException
from app.services.embedding import get_embedding_service
from app.services.llm import get_llm_service
from app.services.vector_store import get_vector_store

logger = logging.getLogger("rag_service")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Lifespan context manager for startup warm-up and graceful shutdown."""
    settings = get_settings()
    logger.info(f"Starting {settings.app_name} in {settings.environment} mode...")

    # Pre-warm Embedding Model singleton
    try:
        logger.info(f"Loading embedding model: {settings.embedding_model_name}")
        get_embedding_service()
        logger.info("Embedding model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load embedding model on startup: {e}")

    # Initialize Vector Store
    try:
        logger.info(f"Initializing ChromaDB vector store at: {settings.chroma_persist_dir}")
        get_vector_store()
        logger.info("ChromaDB vector store initialized successfully.")
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB on startup: {e}")

    # Check Ollama connectivity
    llm_service = get_llm_service()
    is_ollama_online = await llm_service.is_healthy()
    if is_ollama_online:
        logger.info(f"Connected to Ollama at {settings.ollama_base_url} (model: {settings.ollama_model})")
    else:
        logger.warning(
            f"Ollama instance at {settings.ollama_base_url} is unreachable. "
            "Ensure Ollama is running before issuing LLM queries."
        )

    yield

    logger.info("Shutting down application...")


def create_application() -> FastAPI:
    """FastAPI application factory."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        openapi_url=f"{settings.api_v1_str}/openapi.json" if settings.debug else None,
        docs_url=f"{settings.api_v1_str}/docs" if settings.debug else None,
        redoc_url=f"{settings.api_v1_str}/redoc" if settings.debug else None,
        lifespan=lifespan,
    )

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Global Domain Exception Handler
    @app.exception_handler(RAGException)
    async def rag_exception_handler(request: Request, exc: RAGException) -> JSONResponse:
        logger.warning(f"Domain exception [{exc.error_code}]: {exc.message}")
        return JSONResponse(
            status_code=exc.status_code,
            content={
                "error": {
                    "code": exc.error_code,
                    "message": exc.message,
                    "details": exc.details,
                }
            },
        )

    # Unhandled Exception Handler
    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error(f"Unhandled error processing {request.method} {request.url}: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": {
                    "code": "INTERNAL_SERVER_ERROR",
                    "message": "An unexpected internal server error occurred.",
                    "details": {},
                }
            },
        )

    # API Routers
    app.include_router(api_router, prefix=settings.api_v1_str)

    # Root & Health Check Endpoints
    @app.get("/", tags=["health"])
    async def root() -> dict[str, str]:
        return {
            "app_name": settings.app_name,
            "status": "online",
            "docs": f"{settings.api_v1_str}/docs" if settings.debug else "disabled",
        }

    @app.get("/health", tags=["health"])
    async def health_check() -> dict[str, object]:
        llm = get_llm_service()
        ollama_ok = await llm.is_healthy()
        return {
            "status": "healthy" if ollama_ok else "degraded",
            "ollama_connected": ollama_ok,
            "environment": settings.environment,
        }

    return app


app = create_application()
