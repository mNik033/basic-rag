from fastapi import Depends
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.llm import OllamaLLMService, get_llm_service
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager, get_vector_store


def get_rag_service_dep(
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreManager = Depends(get_vector_store),
    llm_service: OllamaLLMService = Depends(get_llm_service),
) -> RAGService:
    return RAGService(
        embedding_service=embedding_service,
        vector_store=vector_store,
        llm_service=llm_service,
    )
