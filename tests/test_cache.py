import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager
from app.domain.schemas import DocumentChunk, ParsedDocument, DocumentMetadata


@pytest.fixture
def mock_embedding_service():
    embedder = MagicMock()
    # Return 384-dimensional dummy vectors
    embedder.embed_text.return_value = [0.1] * 384
    embedder.embed_documents.return_value = [[0.1] * 384]
    return embedder


@pytest.fixture
def mock_llm_service():
    llm = MagicMock()
    llm.model = "gemma4:e2b"
    llm.generate_response = AsyncMock(return_value="This is a test generated answer.")
    return llm


@pytest.fixture
def in_memory_vector_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path / "chroma"))
    monkeypatch.setenv("CHROMA_COLLECTION_NAME", "test_collection")
    monkeypatch.setenv("CACHE_COLLECTION_NAME", "test_cache")
    monkeypatch.setenv("CACHE_SIMILARITY_THRESHOLD", "0.90")
    monkeypatch.setenv("CACHE_ENABLED", "True")
    
    # Reload settings to pick up temp dir
    from app.core.config import get_settings
    get_settings.cache_clear()
    
    store = VectorStoreManager()
    return store


@pytest.mark.asyncio
async def test_semantic_cache_hit_and_miss(mock_embedding_service, mock_llm_service, in_memory_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=in_memory_vector_store,
        llm_service=mock_llm_service,
    )

    # Ingest a dummy chunk
    chunk = DocumentChunk(
        chunk_id="doc1_c1",
        doc_id="doc1",
        content="Revenue for Q3 was 5 million dollars.",
        chunk_index=0,
        metadata={"filename": "report.txt", "doc_id": "doc1"},
    )
    in_memory_vector_store.upsert_chunks([chunk], [[0.1] * 384])

    # First query -> Cache Miss, calls LLM
    res1 = await rag_service.answer_query(query="What was Q3 revenue?", use_cache=True)
    assert res1.cached is False
    assert res1.answer == "This is a test generated answer."
    assert mock_llm_service.generate_response.call_count == 1

    # Second query -> Cache Hit, does NOT call LLM
    res2 = await rag_service.answer_query(query="What was Q3 revenue?", use_cache=True)
    assert res2.cached is True
    assert res2.answer == "This is a test generated answer."
    assert mock_llm_service.generate_response.call_count == 1  # count unchanged!


@pytest.mark.asyncio
async def test_cache_invalidation_on_document_ingest(mock_embedding_service, mock_llm_service, in_memory_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=in_memory_vector_store,
        llm_service=mock_llm_service,
    )

    v1 = in_memory_vector_store.get_collection_version()

    # Ingest document
    chunk1 = DocumentChunk(
        chunk_id="doc1_c1",
        doc_id="doc1",
        content="Initial document version.",
        chunk_index=0,
        metadata={"filename": "doc1.txt", "doc_id": "doc1"},
    )
    in_memory_vector_store.upsert_chunks([chunk1], [[0.1] * 384])
    v2 = in_memory_vector_store.get_collection_version()
    assert v2 > v1

    # Query 1 -> Miss, cached
    res1 = await rag_service.answer_query(query="Tell me about doc1", use_cache=True)
    assert res1.cached is False
    assert mock_llm_service.generate_response.call_count == 1

    # Query 2 -> Hit
    res2 = await rag_service.answer_query(query="Tell me about doc1", use_cache=True)
    assert res2.cached is True
    assert mock_llm_service.generate_response.call_count == 1

    # Ingest new document -> Bumps version and invalidates cache
    chunk2 = DocumentChunk(
        chunk_id="doc2_c1",
        doc_id="doc2",
        content="Second document added.",
        chunk_index=0,
        metadata={"filename": "doc2.txt", "doc_id": "doc2"},
    )
    in_memory_vector_store.upsert_chunks([chunk2], [[0.1] * 384])
    v3 = in_memory_vector_store.get_collection_version()
    assert v3 > v2

    # Query 3 -> Cache Miss because version changed
    res3 = await rag_service.answer_query(query="Tell me about doc1", use_cache=True)
    assert res3.cached is False
    assert mock_llm_service.generate_response.call_count == 2
