from functools import lru_cache
from typing import Literal
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Settings
    app_name: str = "Production RAG Service"
    environment: Literal["development", "staging", "production"] = "development"
    debug: bool = True
    api_v1_str: str = "/api/v1"

    # ChromaDB Settings
    chroma_persist_dir: str = "data/chromadb"
    chroma_collection_name: str = "rag_knowledge_base"

    # Embedding Settings
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384

    # Chunking Defaults
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Semantic Cache Settings
    cache_enabled: bool = True
    cache_collection_name: str = "rag_query_cache"
    cache_similarity_threshold: float = 0.93

    # Context Pruning Settings
    similarity_threshold: float = 0.35

    # LLM Provider (Ollama)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "gemma4:e2b"
    ollama_timeout_seconds: float = 300.0
    ollama_keep_alive: str = "15m"
    ollama_num_predict: int = 512
    ollama_num_ctx: int = 2048
    ollama_temperature: float = 0.2


@lru_cache()
def get_settings() -> Settings:
    return Settings()
