from typing import Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.domain.schemas import DocumentChunk


class VectorStoreManager:
    """Manages persistent ChromaDB client, collection, upserting, querying, and deletion."""

    def __init__(self) -> None:
        settings = get_settings()
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = settings.chroma_collection_name

        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB vector store: {str(e)}") from e

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Upsert document chunks, their embeddings, and metadata into ChromaDB."""
        if not chunks or not embeddings:
            return

        if len(chunks) != len(embeddings):
            raise VectorStoreError("Mismatch between number of chunks and embeddings count.")

        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        try:
            self._collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to upsert chunks into vector store: {str(e)}") from e

    def query(self, query_embedding: list[float], n_results: int = 3) -> dict:
        """Query the vector store for nearest neighbor chunks using cosine similarity."""
        try:
            results = self._collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"],
            )
            return results
        except Exception as e:
            raise VectorStoreError(f"Vector store query failed: {str(e)}") from e

    def list_documents(self) -> list[dict]:
        """List distinct ingested documents by grouping chunk metadata."""
        try:
            data = self._collection.get(include=["metadatas"])
            metadatas = data.get("metadatas", [])
            
            docs_map = {}
            for meta in metadatas:
                if not meta:
                    continue
                doc_id = meta.get("doc_id")
                if doc_id and doc_id not in docs_map:
                    docs_map[doc_id] = {
                        "doc_id": doc_id,
                        "filename": meta.get("filename"),
                        "content_hash": meta.get("content_hash"),
                        "file_type": meta.get("file_type"),
                        "file_size": meta.get("file_size"),
                        "created_at": meta.get("created_at"),
                    }
            return list(docs_map.values())
        except Exception as e:
            raise VectorStoreError(f"Failed to list documents from vector store: {str(e)}") from e

    def delete_document(self, doc_id: str) -> bool:
        """Delete all chunks associated with a given document ID."""
        try:
            data = self._collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            chunk_ids = data.get("ids", [])
            if not chunk_ids:
                return False

            self._collection.delete(ids=chunk_ids)
            return True
        except Exception as e:
            raise VectorStoreError(f"Failed to delete document '{doc_id}': {str(e)}") from e

    def get_by_content_hash(self, content_hash: str) -> Optional[dict]:
        """Check if a document with the given content hash already exists."""
        try:
            data = self._collection.get(
                where={"content_hash": content_hash},
                include=["metadatas"],
                limit=1,
            )
            metadatas = data.get("metadatas", [])
            if metadatas and metadatas[0]:
                return metadatas[0]
            return None
        except Exception as e:
            raise VectorStoreError(f"Failed to check content hash: {str(e)}") from e


def get_vector_store() -> VectorStoreManager:
    """Dependency / accessor function for VectorStoreManager."""
    return VectorStoreManager()
