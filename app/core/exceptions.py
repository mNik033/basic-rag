from typing import Any, Optional


class RAGException(Exception):
    """Base exception for application domain errors."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_SERVER_ERROR",
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}


# --- Not Found Exceptions (404) ---
class NotFoundException(RAGException):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=404,
            error_code="RESOURCE_NOT_FOUND",
            details=details,
        )


class DocumentNotFoundError(NotFoundException):
    def __init__(self, doc_id: str) -> None:
        super().__init__(
            message=f"Document with ID '{doc_id}' was not found.",
            details={"doc_id": doc_id},
        )


# --- Validation Exceptions (400 / 422) ---
class ValidationException(RAGException):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=400,
            error_code="VALIDATION_ERROR",
            details=details,
        )


class UnsupportedFileTypeError(ValidationException):
    def __init__(self, extension: str, supported: list[str]) -> None:
        super().__init__(
            message=f"Unsupported file type '{extension}'. Supported types: {', '.join(supported)}",
            details={"extension": extension, "supported": supported},
        )


class EmptyDocumentError(ValidationException):
    def __init__(self, filename: str) -> None:
        super().__init__(
            message=f"Document '{filename}' contains no extractable text content.",
            details={"filename": filename},
        )


class DuplicateDocumentError(ValidationException):
    def __init__(self, content_hash: str, filename: str) -> None:
        super().__init__(
            message=f"A document with matching content hash '{content_hash}' already exists ('{filename}').",
            details={"content_hash": content_hash, "filename": filename},
        )


# --- Internal Service Exceptions (500) ---
class ServiceException(RAGException):
    def __init__(self, message: str, details: Optional[dict[str, Any]] = None) -> None:
        super().__init__(
            message=message,
            status_code=500,
            error_code="SERVICE_ERROR",
            details=details,
        )


class EmbeddingModelError(ServiceException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Embedding model execution failed: {message}",
            details={"error": message},
        )


class VectorStoreError(ServiceException):
    def __init__(self, message: str) -> None:
        super().__init__(
            message=f"Vector store operation failed: {message}",
            details={"error": message},
        )


# --- External Provider Exceptions (502 / 504) ---
class ExternalProviderException(RAGException):
    def __init__(
        self,
        message: str,
        status_code: int = 502,
        details: Optional[dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="EXTERNAL_PROVIDER_ERROR",
            details=details,
        )


class LLMConnectionError(ExternalProviderException):
    def __init__(self, base_url: str, error: str) -> None:
        super().__init__(
            message=f"Failed to connect to LLM provider at '{base_url}': {error}",
            status_code=502,
            details={"base_url": base_url, "error": error},
        )


class LLMResponseError(ExternalProviderException):
    def __init__(self, status_code: int, message: str) -> None:
        super().__init__(
            message=f"LLM provider returned error {status_code}: {message}",
            status_code=502,
            details={"status_code": status_code, "error": message},
        )
