import json
import pytest
from unittest.mock import MagicMock
from app.services.rag_service import RAGService, NO_INFO_FALLBACK


@pytest.fixture
def mock_embedding_service():
    embedder = MagicMock()
    embedder.embed_text.return_value = [0.1] * 384
    return embedder


@pytest.fixture
def mock_llm_service():
    llm = MagicMock()
    llm.model = "gemma4:e2b"

    async def fake_stream(prompt, system_prompt=None):
        tokens = ["Hello", " world", "!"]
        for t in tokens:
            yield t

    llm.stream_response = fake_stream
    return llm


@pytest.fixture
def mock_vector_store():
    store = MagicMock()
    store.query_cache.return_value = None
    return store


@pytest.mark.asyncio
async def test_stream_query_full_flow(mock_embedding_service, mock_llm_service, mock_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_service=mock_llm_service,
    )

    mock_vector_store.query.return_value = {
        "documents": [["Valid context passage."]],
        "metadatas": [[{"filename": "doc.txt"}]],
        "ids": [["c1"]],
        "distances": [[0.1]],  # sim = 0.9 >= 0.35
    }

    events = []
    async for event_line in rag_service.stream_query(query="Tell me something", use_cache=True):
        if event_line.startswith("data: "):
            payload = json.loads(event_line.replace("data: ", "").strip())
            events.append(payload)

    # Verify event structure
    assert len(events) >= 5  # start, "Hello", " world", "!", end
    assert events[0]["event"] == "start"
    assert events[0]["cached"] is False
    assert len(events[0]["sources"]) == 1

    tokens = [e["content"] for e in events if e.get("event") == "token"]
    assert "".join(tokens) == "Hello world!"

    assert events[-1]["event"] == "end"
    assert events[-1]["cached"] is False

    # Verify that full answer was cached
    mock_vector_store.store_query_cache.assert_called_once()
    saved_answer = mock_vector_store.store_query_cache.call_args[1]["answer"]
    assert saved_answer == "Hello world!"


@pytest.mark.asyncio
async def test_stream_query_cache_hit(mock_embedding_service, mock_llm_service, mock_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_service=mock_llm_service,
    )

    mock_vector_store.query_cache.return_value = {
        "answer": "Cached answer streaming response",
        "sources": [{"chunk_id": "c1", "content": "data"}],
        "model": "gemma4:e2b",
        "similarity": 0.98,
    }

    events = []
    async for event_line in rag_service.stream_query(query="Cached question"):
        if event_line.startswith("data: "):
            payload = json.loads(event_line.replace("data: ", "").strip())
            events.append(payload)

    assert events[0]["event"] == "start"
    assert events[0]["cached"] is True
    assert events[1]["event"] == "token"
    assert events[1]["content"] == "Cached answer streaming response"
    assert events[2]["event"] == "end"
    assert events[2]["cached"] is True


@pytest.mark.asyncio
async def test_stream_query_early_exit(mock_embedding_service, mock_llm_service, mock_vector_store):
    rag_service = RAGService(
        embedding_service=mock_embedding_service,
        vector_store=mock_vector_store,
        llm_service=mock_llm_service,
    )

    # All chunks low similarity
    mock_vector_store.query.return_value = {
        "documents": [["Unrelated text"]],
        "metadatas": [[{}]],
        "ids": [["c1"]],
        "distances": [[0.95]],  # sim = 0.05 < 0.35
    }

    events = []
    async for event_line in rag_service.stream_query(query="Unrelated question"):
        if event_line.startswith("data: "):
            payload = json.loads(event_line.replace("data: ", "").strip())
            events.append(payload)

    assert events[0]["event"] == "start"
    assert events[0]["model"] == "fast-path-early-exit"
    assert events[1]["event"] == "token"
    assert events[1]["content"] == NO_INFO_FALLBACK
    assert events[2]["event"] == "end"
