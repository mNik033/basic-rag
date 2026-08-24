import logging
from typing import List, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from app.domain.knowledge import (
    EngineeringDocument,
    KnowledgeIndexRequest,
    KnowledgeIndexResponse,
    KnowledgeStatusResponse,
)
from app.domain.models import PRUnderstanding, PullRequest, Repository
from app.domain.schemas import DocumentChunk
from app.services.embedding import EmbeddingService, get_embedding_service
from app.services.github.knowledge_builder import EngineeringDocumentSynthesizer
from app.services.vector_store import VectorStoreManager, get_vector_store

logger = logging.getLogger(__name__)


class KnowledgeBaseService:
    """Coordinates synthesizing engineering documents, generating vector embeddings, and indexing in ChromaDB."""

    def __init__(
        self,
        session: AsyncSession,
        embedding_service: Optional[EmbeddingService] = None,
        vector_store: Optional[VectorStoreManager] = None,
    ) -> None:
        self.session = session
        self.embedding_service = embedding_service or get_embedding_service()
        self.vector_store = vector_store or get_vector_store()

    async def get_repository(
        self, repository_id: Optional[int] = None, owner: Optional[str] = None, repo: Optional[str] = None
    ) -> Repository:
        """Find repository by id or (owner, repo)."""
        stmt = select(Repository)
        if repository_id:
            stmt = stmt.where(Repository.id == repository_id)
        elif owner and repo:
            stmt = stmt.where(Repository.owner == owner, Repository.name == repo)
        else:
            raise ValueError("Must provide either repository_id or (owner, repo).")

        res = await self.session.execute(stmt)
        record = res.scalar_one_or_none()
        if not record:
            raise ValueError("Repository not found.")
        return record

    async def get_knowledge_status(self, owner: str, repo: str) -> KnowledgeStatusResponse:
        """Get summary of total PRs, analyzed PRs, and indexed vector count for a repo."""
        repo_obj = await self.get_repository(owner=owner, repo=repo)

        # Count total PRs
        pr_stmt = select(PullRequest).where(PullRequest.repository_id == repo_obj.id)
        total_prs = len((await self.session.execute(pr_stmt)).scalars().all())

        # Count understood PRs
        u_stmt = (
            select(PRUnderstanding)
            .join(PullRequest, PRUnderstanding.pull_request_id == PullRequest.id)
            .where(PullRequest.repository_id == repo_obj.id)
        )
        understood_prs = len((await self.session.execute(u_stmt)).scalars().all())

        # Count indexed vectors from vector store
        repo_full_name = f"{owner}/{repo}"
        indexed_count = 0
        try:
            stored = self.vector_store.collection.get(
                where={"repository": repo_full_name},
                include=["metadatas"],
            )
            indexed_count = len(stored.get("ids", []))
        except Exception as e:
            logger.warning("Could not query vector store count for %s: %s", repo_full_name, e)

        return KnowledgeStatusResponse(
            repository=repo_full_name,
            total_prs=total_prs,
            understood_prs=understood_prs,
            indexed_vectors=indexed_count,
        )

    async def index_repository_knowledge(
        self, request: KnowledgeIndexRequest
    ) -> KnowledgeIndexResponse:
        """Synthesize and index engineering documents and embeddings into vector store."""
        repo = await self.get_repository(
            repository_id=request.repository_id, owner=request.owner, repo=request.repo
        )

        query = (
            select(PullRequest)
            .where(PullRequest.repository_id == repo.id)
            .options(
                selectinload(PullRequest.commits),
                selectinload(PullRequest.changed_files),
                selectinload(PullRequest.understanding),
            )
            .order_by(PullRequest.number.asc())
        )

        if request.pr_number:
            query = query.where(PullRequest.number == request.pr_number)

        if request.limit:
            query = query.limit(request.limit)

        prs = (await self.session.execute(query)).scalars().all()
        logger.info("Found %d PRs for vector indexing in %s/%s.", len(prs), repo.owner, repo.name)

        if not prs:
            return KnowledgeIndexResponse(
                repository=f"{repo.owner}/{repo.name}",
                status="completed",
                documents_indexed=0,
                chunks_created=0,
                message="No matching pull requests found for indexing.",
            )

        # 1. Synthesize documents
        synthesizer = EngineeringDocumentSynthesizer()
        documents: List[EngineeringDocument] = []
        for pr in prs:
            doc = synthesizer.synthesize_pr_document(
                repo=repo,
                pr=pr,
                understanding=pr.understanding,
            )
            documents.append(doc)

        # 2. Check existing indexed documents to avoid re-embedding if not force_reindex
        docs_to_index: List[EngineeringDocument] = []
        if not request.force_reindex:
            for doc in documents:
                existing = self.vector_store.collection.get(ids=[doc.doc_id])
                if not existing or len(existing.get("ids", [])) == 0:
                    docs_to_index.append(doc)
                else:
                    logger.debug("Document %s already indexed in vector store; skipping.", doc.doc_id)
        else:
            docs_to_index = documents

        if not docs_to_index:
            return KnowledgeIndexResponse(
                repository=f"{repo.owner}/{repo.name}",
                status="completed",
                documents_indexed=0,
                chunks_created=0,
                message=f"All {len(documents)} PR documents are already indexed in vector store.",
            )

        # 3. Batch generate dense embeddings
        texts = [doc.text for doc in docs_to_index]
        logger.info("Generating %d dense embeddings using %s...", len(texts), self.embedding_service.model_name)
        embeddings = self.embedding_service.embed_documents(texts)

        # 4. Construct DocumentChunk items and upsert to ChromaDB
        chunks: List[DocumentChunk] = []
        for i, doc in enumerate(docs_to_index):
            chunk = DocumentChunk(
                chunk_id=doc.doc_id,
                doc_id=doc.doc_id,
                content=doc.text,
                chunk_index=0,
                metadata=doc.metadata,
            )
            chunks.append(chunk)

        self.vector_store.upsert_chunks(chunks=chunks, embeddings=embeddings)

        logger.info(
            "Successfully indexed %d engineering documents into vector store for %s/%s.",
            len(chunks),
            repo.owner,
            repo.name,
        )

        return KnowledgeIndexResponse(
            repository=f"{repo.owner}/{repo.name}",
            status="completed",
            documents_indexed=len(chunks),
            chunks_created=len(chunks),
            message=f"Successfully indexed {len(chunks)} engineering documents into vector store.",
        )
