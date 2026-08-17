import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.rag_service import RAGService, NO_INFO_FALLBACK
from app.domain.schemas import DocumentChunk


@pytest.fixture
def mock_embedding_service():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_llm_service():
    llm = MagicMock()
    llm.model = "gemma4:e2b"
    llm.generate_response = AsyncMock(return_value="Detailed answer about topic.")
    return llm


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query_cache.return_value = None
    return store


@pytest.mark.asyncio
async def test_early_exit_when_no_chunks_meet_threshold(mock_embedding_service, mock_llm_service, mock_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_service=mock_llm_service,
    )

    # Mock ChromaDB query returning low-similarity chunks (distance = 0.9 => similarity = 0.1)
    mock_vector_store.query.return_value = {
        "documents": [["Irrelevant passage about unrelated topic."]],
        "metadatas": [[{"filename": "random.txt"}]],
        "ids": [["chunk_0"]],
        "distances": [[0.9]],  # sim = 0.1 < 0.35
    }

    result = await rag_service.answer_query(
        query="What is the quantum state of electrons?",
        similarity_threshold=0.35,
    )

    assert result.answer == NO_INFO_FALLBACK
    assert result.sources == []
    assert result.model == "fast-path-early-exit"
    # Verify LLM was NOT called
    mock_llm_service.generate_response.assert_not_called()


@pytest.mark.asyncio
async def test_context_pruning_filters_irrelevant_chunks(mock_embedding_service, mock_llm_service, mock_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_service=mock_llm_service,
    )

    # 1 relevant chunk (dist 0.2 => sim 0.8), 1 irrelevant chunk (dist 0.8 => sim 0.2)
    mock_vector_store.query.return_value = {
        "documents": [["Relevant passage on revenue.", "Random noise about weather."]],
        "metadatas": [[{"filename": "finance.txt"}, {"filename": "weather.txt"}]],
        "ids": [["c1", "c2"]],
        "distances": [[0.2, 0.8]],
    }

    result = await rag_service.answer_query(
        query="What was revenue?",
        similarity_threshold=0.35,
    )

    assert result.answer == "Detailed answer about topic."
    assert len(result.sources) == 1
    assert result.sources[0].chunk_id == "c1"
    mock_llm_service.generate_response.assert_called_once()
    
    # Check that prompt only contained the relevant chunk
    call_prompt = mock_llm_service.generate_response.call_args[1]["prompt"]
    assert "Relevant passage on revenue." in call_prompt
    assert "Random noise about weather." not in call_prompt
