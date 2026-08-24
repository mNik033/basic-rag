from functools import lru_cache
from typing import Literal, Optional
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
    ollama_num_predict: int = 1536
    ollama_num_ctx: int = 4096
    ollama_temperature: float = 0.1
    ollama_repeat_penalty: float = 1.2
    ollama_repeat_last_n: int = 64
    ollama_top_k: int = 40
    ollama_top_p: float = 0.9
    ollama_think: bool = False
    ollama_num_thread: Optional[int] = None

    # PostgreSQL Database Settings
    postgres_user: str = "postgres"
    postgres_password: str = "postgres"
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "engineering_memory"
    database_url: Optional[str] = None
    db_echo: bool = False

    @property
    def async_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        return f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"

    # GitHub Data Collector Settings
    github_token: Optional[str] = None
    github_api_base_url: str = "https://api.github.com"
    github_rate_limit_pause_seconds: int = 60
    github_max_retries: int = 3
    github_per_page: int = 50


@lru_cache()
def get_settings() -> Settings:
    return Settings()
