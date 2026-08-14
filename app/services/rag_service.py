from typing import Optional
from app.core.exceptions import DuplicateDocumentError, EmptyDocumentError
from app.domain.schemas import ParsedDocument, RAGQueryResult, RetrievedChunk
from app.services.chunking import RecursiveCharacterTextSplitter
from app.services.document_parser import DocumentParser
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.llm import OllamaLLMService, get_llm_service
from app.services.vector_store import VectorStoreManager, get_vector_store

SYSTEM_PROMPT = """You are a knowledgeable and concise AI assistant.
Answer the question strictly based on the provided context passages.
If the context does not contain enough information to answer the question, clearly state: "I don't have enough information in the provided documents to answer that question."
Always cite or refer to the relevant source documents when possible.
"""


class RAGService:
    """Orchestrates document ingestion and retrieval-augmented question answering."""

    def __init__(
        self,
        parser: Optional[DocumentParser] = None,
        splitter: Optional[RecursiveCharacterTextSplitter] = None,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreManager] = None,
        llm_service: Optional[OllamaLLMService] = None,
    ) -> None:
        self.parser = parser or DocumentParser()
        self.splitter = splitter or RecursiveCharacterTextSplitter()
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()
        self.llm_service = llm_service or get_llm_service()

    def ingest_document(self, filename: str, content_bytes: bytes) -> ParsedDocument:
        """Parse raw file bytes, split into chunks, embed, and store in vector database."""
        content_hash = self.parser.compute_hash(content_bytes)
        existing = self.vector_store.get_by_content_hash(content_hash)
        if existing:
            raise DuplicateDocumentError(
                content_hash=content_hash,
                filename=existing.get("filename", filename),
            )

        parsed_doc = self.parser.parse(filename, content_bytes)
        chunks = self.splitter.split_document(parsed_doc)

        if not chunks:
            raise EmptyDocumentError(filename=filename)

        texts = [chunk.content for chunk in chunks]
        embeddings = self.embedding_service.embed_documents(texts)
        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        return parsed_doc

    async def answer_query(
        self,
        query: str,
        n_results: int = 3,
        system_prompt: Optional[str] = None,
    ) -> RAGQueryResult:
        """Perform semantic search, build augmented context prompt, and generate LLM answer."""
        query_embedding = self.embedding_service.embed_text(query)
        search_results = self.vector_store.query(query_embedding, n_results=n_results)

        retrieved_chunks: list[RetrievedChunk] = []
        documents = search_results.get("documents", [[]])[0]
        metadatas = search_results.get("metadatas", [[]])[0]
        ids = search_results.get("ids", [[]])[0]
        distances = search_results.get("distances", [[]])[0]

        for i, doc_text in enumerate(documents):
            chunk_id = ids[i] if i < len(ids) else f"chunk_{i}"
            meta = metadatas[i] if i < len(metadatas) else {}
            dist = distances[i] if i < len(distances) else None
            sim_score = (1.0 - dist) if dist is not None else None

            retrieved_chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    content=doc_text,
                    metadata=meta,
                    similarity_score=sim_score,
                )
            )

        context_blocks = []
        for i, chunk in enumerate(retrieved_chunks, start=1):
            source = chunk.metadata.get("filename", "unknown")
            context_blocks.append(f"[Source {i}: {source}]\n{chunk.content}")

        formatted_context = "\n\n".join(context_blocks)
        prompt = (
            f"Context Information:\n"
            f"---------------------\n"
            f"{formatted_context}\n"
            f"---------------------\n\n"
            f"Question: {query}\n"
            f"Answer:"
        )

        answer = await self.llm_service.generate_response(
            prompt=prompt,
            system_prompt=system_prompt or SYSTEM_PROMPT,
        )

        return RAGQueryResult(
            query=query,
            answer=answer,
            sources=retrieved_chunks,
            model=self.llm_service.model,
        )


def get_rag_service() -> RAGService:
    """Dependency / accessor function for RAGService."""
    return RAGService()
