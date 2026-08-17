from fastapi import APIRouter, Depends, File, UploadFile, status
from app.api.dependencies import get_rag_service_dep
from app.core.exceptions import DocumentNotFoundError
from app.domain.schemas import (
    DocumentDeleteResponse,
    DocumentResponse,
    DocumentUploadResponse,
)
from app.services.rag_service import RAGService
from app.services.vector_store import VectorStoreManager, get_vector_store

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload and ingest a document",
)
async def upload_document(
    file: UploadFile = File(...),
    rag_service: RAGService = Depends(get_rag_service_dep),
) -> DocumentUploadResponse:
    content_bytes = await file.read()
    parsed_doc = await rag_service.ingest_document(
        filename=file.filename or "unknown",
        content_bytes=content_bytes,
    )

    chunks_created = len(rag_service.splitter.split_document(parsed_doc))

    return DocumentUploadResponse(
        message="Document successfully parsed, embedded, and stored.",
        doc_id=parsed_doc.doc_id,
        filename=parsed_doc.metadata.filename,
        chunks_created=chunks_created,
    )


@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="List all ingested documents",
)
def list_documents(
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> list[DocumentResponse]:
    docs = vector_store.list_documents()
    return [
        DocumentResponse(
            doc_id=str(d.get("doc_id", "")),
            filename=str(d.get("filename", "")),
            content_hash=str(d.get("content_hash", "")),
            file_type=str(d.get("file_type", "")),
            file_size=int(d.get("file_size", 0)),
            created_at=str(d.get("created_at", "")),
        )
        for d in docs
    ]


@router.delete(
    "/{doc_id}",
    response_model=DocumentDeleteResponse,
    summary="Delete a document and its chunks by ID",
)
def delete_document(
    doc_id: str,
    vector_store: VectorStoreManager = Depends(get_vector_store),
) -> DocumentDeleteResponse:
    deleted = vector_store.delete_document(doc_id)
    if not deleted:
        raise DocumentNotFoundError(doc_id=doc_id)
    return DocumentDeleteResponse(
        message=f"Document '{doc_id}' and all associated chunks deleted.",
        doc_id=doc_id,
    )
