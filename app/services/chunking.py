import uuid
from typing import Optional
from app.core.config import get_settings
from app.domain.schemas import DocumentChunk, ParsedDocument


class RecursiveCharacterTextSplitter:
    """Recursively splits text into chunks using a hierarchical list of separators

    to maintain semantic coherence while respecting chunk size and overlap limits.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", "? ", "! ", " ", ""]

    def __init__(
        self,
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
        separators: Optional[list[str]] = None,
    ) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size if chunk_size is not None else settings.chunk_size
        self.chunk_overlap = chunk_overlap if chunk_overlap is not None else settings.chunk_overlap
        self.separators = separators or self.DEFAULT_SEPARATORS

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be smaller than chunk_size ({self.chunk_size})"
            )

    def split_text(self, text: str) -> list[str]:
        """Split a raw text string into a list of chunks."""
        return self._split(text, self.separators)

    def split_document(self, document: ParsedDocument) -> list[DocumentChunk]:
        """Split a ParsedDocument into DocumentChunk instances with preserved metadata."""
        raw_chunks = self.split_text(document.text)
        chunks: list[DocumentChunk] = []

        for idx, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{document.doc_id}_chunk_{idx}"
            chunk_metadata = {
                "doc_id": document.doc_id,
                "filename": document.metadata.filename,
                "content_hash": document.metadata.content_hash,
                "file_type": document.metadata.file_type,
                "file_size": document.metadata.file_size,
                "chunk_index": idx,
                "total_chunks": len(raw_chunks),
                "created_at": document.metadata.created_at.isoformat(),
            }
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    doc_id=document.doc_id,
                    content=chunk_text,
                    chunk_index=idx,
                    metadata=chunk_metadata,
                )
            )

        return chunks

    def _split(self, text: str, separators: list[str]) -> list[str]:
        if not text:
            return []

        if len(text) <= self.chunk_size:
            return [text]

        separator = ""
        remaining_separators: list[str] = []
        for i, sep in enumerate(separators):
            if sep == "":
                separator = ""
                remaining_separators = []
                break
            if sep in text:
                separator = sep
                remaining_separators = separators[i + 1 :]
                break

        splits = self._split_by_separator(text, separator)

        good_splits: list[str] = []
        for piece in splits:
            if len(piece) <= self.chunk_size:
                good_splits.append(piece)
            else:
                if remaining_separators:
                    sub_splits = self._split(piece, remaining_separators)
                    good_splits.extend(sub_splits)
                else:
                    good_splits.extend(self._hard_split(piece, self.chunk_size, self.chunk_overlap))

        return self._merge_splits(good_splits, separator)

    def _split_by_separator(self, text: str, separator: str) -> list[str]:
        if not separator:
            return list(text)
        return text.split(separator)

    def _hard_split(self, text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
        step = chunk_size - chunk_overlap
        if step <= 0:
            step = 1
        return [text[i : i + chunk_size] for i in range(0, len(text), step)]

    def _merge_splits(self, splits: list[str], separator: str) -> list[str]:
        chunks: list[str] = []
        current_chunk: list[str] = []
        current_length = 0

        for piece in splits:
            piece_len = len(piece)
            sep_len = len(separator) if current_chunk else 0

            if current_length + sep_len + piece_len <= self.chunk_size:
                current_chunk.append(piece)
                current_length += sep_len + piece_len
            else:
                if current_chunk:
                    merged = separator.join(current_chunk).strip()
                    if merged:
                        chunks.append(merged)

                    overlap_chunk: list[str] = []
                    overlap_length = 0
                    for prev_piece in reversed(current_chunk):
                        prev_sep_len = len(separator) if overlap_chunk else 0
                        if overlap_length + prev_sep_len + len(prev_piece) <= self.chunk_overlap:
                            overlap_chunk.insert(0, prev_piece)
                            overlap_length += prev_sep_len + len(prev_piece)
                        else:
                            break
                    current_chunk = overlap_chunk
                    current_length = sum(len(p) for p in current_chunk) + (
                        len(separator) * max(0, len(current_chunk) - 1)
                    )

                current_chunk.append(piece)
                current_length += (len(separator) if len(current_chunk) > 1 else 0) + piece_len

        if current_chunk:
            merged = separator.join(current_chunk).strip()
            if merged:
                chunks.append(merged)

        return chunks
