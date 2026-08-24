import json
import uuid
from datetime import datetime, timezone
from typing import Any, Optional
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.core.config import get_settings
from app.core.exceptions import VectorStoreError
from app.domain.schemas import DocumentChunk


class VectorStoreManager:
    """Manages persistent ChromaDB client, collection, semantic cache, upserting, querying, and deletion."""

    def __init__(self) -> None:
        settings = get_settings()
        self.persist_dir = settings.chroma_persist_dir
        self.collection_name = settings.chroma_collection_name
        self.cache_collection_name = settings.cache_collection_name
        self.cache_enabled = settings.cache_enabled
        self.cache_threshold = settings.cache_similarity_threshold

        try:
            self._client = chromadb.PersistentClient(
                path=self.persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._cache_collection = self._client.get_or_create_collection(
                name=self.cache_collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._meta_collection = self._client.get_or_create_collection(
                name="rag_system_metadata",
            )
        except Exception as e:
            raise VectorStoreError(f"Failed to initialize ChromaDB vector store: {str(e)}") from e

    @property
    def collection(self):
        """Underlying ChromaDB collection."""
        return self._collection

    def get_collection_version(self) -> int:
        """Get the current knowledge base collection version used for cache validity."""
        try:
            res = self._meta_collection.get(ids=["collection_version"], include=["metadatas"])
            metadatas = res.get("metadatas", [])
            if metadatas and metadatas[0] and "version" in metadatas[0]:
                return int(metadatas[0]["version"])
            
            # Initialize version if not present
            self._meta_collection.upsert(
                ids=["collection_version"],
                documents=["Document collection version counter"],
                metadatas=[{"version": 1, "updated_at": datetime.now(timezone.utc).isoformat()}],
            )
            return 1
        except Exception as e:
            # Fallback default
            return 1

    def increment_collection_version(self) -> int:
        """Increment the knowledge base collection version to invalidate outdated query caches."""
        try:
            current_version = self.get_collection_version()
            new_version = current_version + 1
            self._meta_collection.upsert(
                ids=["collection_version"],
                documents=["Document collection version counter"],
                metadatas=[{"version": new_version, "updated_at": datetime.now(timezone.utc).isoformat()}],
            )
            return new_version
        except Exception as e:
            raise VectorStoreError(f"Failed to increment collection version: {str(e)}") from e

    def upsert_chunks(self, chunks: list[DocumentChunk], embeddings: list[list[float]]) -> None:
        """Upsert document chunks, their embeddings, and metadata into ChromaDB, and bump collection version."""
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
            self.increment_collection_version()
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

    def query_cache(
        self,
        query_embedding: list[float],
        threshold: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """Check if a semantically similar query exists in cache for the current collection version."""
        if not self.cache_enabled:
            return None

        sim_threshold = threshold if threshold is not None else self.cache_threshold
        current_version = self.get_collection_version()

        try:
            results = self._cache_collection.query(
                query_embeddings=[query_embedding],
                n_results=1,
                where={"version": current_version},
                include=["documents", "metadatas", "distances"],
            )

            docs = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            if not docs or not distances:
                return None

            dist = distances[0]
            similarity = 1.0 - dist if dist is not None else 0.0

            if similarity >= sim_threshold:
                meta = metadatas[0] if metadatas else {}
                sources_raw = meta.get("sources_json", "[]")
                try:
                    sources = json.loads(sources_raw)
                except Exception:
                    sources = []

                return {
                    "answer": docs[0],
                    "sources": sources,
                    "model": meta.get("model", "cached"),
                    "similarity": similarity,
                    "cached_query": meta.get("query", ""),
                    "version": current_version,
                }
            return None
        except Exception:
            # On cache error, gracefully degrade to cache miss
            return None

    def store_query_cache(
        self,
        query: str,
        query_embedding: list[float],
        answer: str,
        sources: list[dict[str, Any]],
        model: str,
    ) -> None:
        """Store a completed query and its generated answer in the semantic vector cache."""
        if not self.cache_enabled or not answer:
            return

        current_version = self.get_collection_version()
        cache_id = str(uuid.uuid4())

        try:
            self._cache_collection.upsert(
                ids=[cache_id],
                embeddings=[query_embedding],
                documents=[answer],
                metadatas=[{
                    "query": query,
                    "version": current_version,
                    "model": model,
                    "sources_json": json.dumps(sources),
                    "created_at": datetime.now(timezone.utc).isoformat(),
                }],
            )
        except Exception:
            # Cache failure shouldn't fail the user query
            pass

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
        """Delete all chunks associated with a given document ID and bump collection version."""
        try:
            data = self._collection.get(
                where={"doc_id": doc_id},
                include=[],
            )
            chunk_ids = data.get("ids", [])
            if not chunk_ids:
                return False

            self._collection.delete(ids=chunk_ids)
            self.increment_collection_version()
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


_vector_store_instance: Optional[VectorStoreManager] = None


def get_vector_store() -> VectorStoreManager:
    """Dependency / accessor function returning a singleton VectorStoreManager."""
    global _vector_store_instance
    if _vector_store_instance is None:
        _vector_store_instance = VectorStoreManager()
    return _vector_store_instance
