"""Configurable sliding-window text chunker with sentence boundary protection."""

import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TextChunk:
    """Represent a single text chunk produced by the chunker."""

    chunk_id: str
    text: str
    start_char: int
    end_char: int
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Sliding window text chunker with overlap and boundary snapping."""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
        min_chunk_size: int = 50,
    ) -> None:
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size

    def chunk_text(
        self,
        text: str,
        doc_id: str = "doc",
        metadata: dict[str, Any] | None = None,
    ) -> list[TextChunk]:
        """Split text string into TextChunk objects with overlap."""
        if not text or not text.strip():
            return []

        base_metadata = metadata or {}
        cleaned_text = text.replace("\r\n", "\n")
        total_len = len(cleaned_text)
        chunks: list[TextChunk] = []

        start = 0
        chunk_idx = 0

        while start < total_len:
            end = start + self.chunk_size

            # If not at the end of text, snap end to nearest whitespace/punctuation boundary
            if end < total_len:
                boundary = self._find_boundary(cleaned_text, start, end)
                if boundary > start + self.min_chunk_size:
                    end = boundary

            chunk_str = cleaned_text[start:end].strip()

            if len(chunk_str) >= self.min_chunk_size or not chunks:
                chunk_meta = base_metadata.copy()
                chunk_meta.update({
                    "chunk_index": chunk_idx,
                    "doc_id": doc_id,
                })
                chunks.append(
                    TextChunk(
                        chunk_id=f"{doc_id}_chunk_{chunk_idx}",
                        text=chunk_str,
                        start_char=start,
                        end_char=end,
                        metadata=chunk_meta,
                    )
                )
                chunk_idx += 1

            # Move window start by stride
            stride = max(1, (end - start) - self.chunk_overlap)
            start += stride

        return chunks

    @staticmethod
    def _find_boundary(text: str, start: int, end: int) -> int:
        """Find boundary position near target end index (sentence or paragraph break)."""
        lookback_window = text[max(start, end - 80) : end]
        # Match paragraph break, period, newline, or sentence end mark
        match = re.search(r"(\.\s+|\n\n|\n|\?\s+|\!\s+)", lookback_window[::-1])
        if match:
            # Calculate actual position from end
            offset = match.start()
            return end - offset
        return end
