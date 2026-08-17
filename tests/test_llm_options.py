import pytest
from unittest.mock import AsyncMock, MagicMock
from app.services.llm import OllamaLLMService


def test_ollama_llm_options_and_keepalive(monkeypatch):
    monkeypatch.setenv("OLLAMA_KEEP_ALIVE", "20m")
    monkeypatch.setenv("OLLAMA_NUM_PREDICT", "256")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "1024")
    monkeypatch.setenv("OLLAMA_TEMPERATURE", "0.1")
    monkeypatch.setenv("OLLAMA_THINK", "False")
    monkeypatch.setenv("OLLAMA_NUM_THREAD", "8")

    from app.core.config import get_settings
    get_settings.cache_clear()

    llm = OllamaLLMService()
    assert llm.keep_alive == "20m"
    assert llm.num_predict == 256
    assert llm.num_ctx == 1024
    assert llm.temperature == 0.1
    assert llm.think is False
    assert llm.num_thread == 8

    options = llm._build_options()
    assert options == {
        "num_predict": 256,
        "num_ctx": 1024,
        "temperature": 0.1,
        "num_thread": 8,
    }
