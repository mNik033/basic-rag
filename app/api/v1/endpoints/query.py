from fastapi import APIRouter, Depends
from app.api.dependencies import get_rag_service_dep
from app.domain.schemas import (
    QueryRequest,
    RAGQueryResult,
    RetrievedChunk,
    SearchRequest,
    SearchResponse,
)
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager, get_vector_store

router = APIRouter(prefix="/query", tags=["query"])


@router.post(
    "/",
    response_model=RAGQueryResult,
    summary="Ask a question using RAG (Retrieve -> Augment -> Generate)",
)
async def query_rag(
    request: QueryRequest,
    rag_service: RAGService = Depends(get_rag_service_dep),
) -> RAGQueryResult:
    return await rag_service.answer_query(
        query=request.query,
        n_results=request.n_results,
        system_prompt=request.system_prompt,
    )


@router.post(
    "/search",
    response_model=SearchResponse,
    summary="Perform pure semantic vector search without LLM generation",
)
def semantic_search(
    request: SearchRequest,
    embedding_service: EmbeddingService = Depends(get_embedding_service),
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> SearchResponse:
    query_vector = embedding_service.embed_text(request.query)
    raw_results = vector_store.query(query_vector, n_results=request.n_results)

    chunks: list[RetrievedChunk] = []
    documents = raw_results.get("documents", [[]])[0]
    metadatas = raw_results.get("metadatas", [[]])[0]
    ids = raw_results.get("ids", [[]])[0]
    distances = raw_results.get("distances", [[]])[0]

    for i, doc_text in enumerate(documents):
        chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
        meta = metadatas[i] if i < len(metadatas) else {}
        dist = distances[i] if i < len(distances) else None
        sim_score = (1.0 - dist) if dist is not None else None

        chunks.append(
            RetrievedChunk(
                chunk_id=chunk_id,
                content=doc_text,
                metadata=meta,
                similarity_score=sim_score,
            )
        )

    return SearchResponse(
        query=request.query,
        total_results=len(chunks),
        results=chunks,
    )
