import os
import threading
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer
from app.core.config import get_settings
from app.core.exceptions import EmbeddingModelError


class EmbeddingService:
    """Singleton service for generating dense vector embeddings using SentenceTransformers."""

    _instance: Optional["EmbeddingService"] = None
    _lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "EmbeddingService":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return

        settings = get_settings()
        self.model_name = settings.embedding_model_name
        self.dimension = settings.embedding_dim

        model_path = str(Path(self.model_name).expanduser())

        try:
            self._model = SentenceTransformer(model_path)
            self._initialized = True
        except Exception as e:
            raise EmbeddingModelError(f"Failed to load embedding model '{model_path}': {str(e)}") from e

    def embed_text(self, text: str) -> list[float]:
        """Generate vector embedding for a single text query or string."""
        if not text or not text.strip():
            raise EmbeddingModelError("Cannot generate embedding for empty text.")

        try:
            embedding = self._model.encode(text, convert_to_numpy=True, normalize_embeddings=True)
            return embedding.tolist()
        except Exception as e:
            raise EmbeddingModelError(f"Embedding generation failed: {str(e)}") from e

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Generate batch vector embeddings for a list of document chunk strings."""
        if not texts:
            return []

        try:
            embeddings = self._model.encode(
                texts,
                batch_size=32,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return embeddings.tolist()
        except Exception as e:
            raise EmbeddingModelError(f"Batch embedding generation failed: {str(e)}") from e


def get_embedding_service() -> EmbeddingService:
    """Dependency / accessor function to retrieve the EmbeddingService singleton."""
    return EmbeddingService()
