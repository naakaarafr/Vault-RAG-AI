"""Structure-aware document chunker with heading splits and token budget overlap limits."""

import re
from dataclasses import dataclass, field
from typing import Any

from vault.ingest.loaders import ExtractedPage


@dataclass
class IngestedChunk:
    """Standardized ingested document chunk contract."""

    chunk_id: str
    doc_id: str
    title: str
    page: int
    allowed_roles: list[str]
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class StructureAwareChunker:
    """Structure-aware chunker splitting on clauses/headings then token budgets."""

    # Heading and clause structure pattern
    STRUCTURE_PATTERN = r"(?m)^(?:#+\s+.+|(?:Section|Clause|Article|\d+\.\d+)\s+.+)$"

    def __init__(
        self,
        max_tokens: int = 400,
        token_overlap: int = 60,
    ) -> None:
        if token_overlap >= max_tokens:
            raise ValueError("token_overlap must be strictly less than max_tokens")
        self.max_tokens = max_tokens
        self.token_overlap = token_overlap

    def chunk_document(
        self,
        doc_id: str,
        title: str,
        allowed_roles: list[str],
        pages: list[ExtractedPage],
    ) -> list[IngestedChunk]:
        """Split document pages into structure-aware IngestedChunk objects."""
        all_chunks: list[IngestedChunk] = []
        global_chunk_counter = 0

        for page in pages:
            if not page.text.strip():
                continue

            # 1. Structure boundary splitting (headings/clauses)
            sections = self._split_by_structure(page.text)

            for section_text in sections:
                if not section_text.strip():
                    continue

                # 2. Token budget sliding window sub-splitting
                token_chunks = self._split_by_token_budget(section_text)

                for chunk_text in token_chunks:
                    chunk_id = f"{doc_id}_p{page.page_number}_c{global_chunk_counter}"
                    all_chunks.append(
                        IngestedChunk(
                            chunk_id=chunk_id,
                            doc_id=doc_id,
                            title=title,
                            page=page.page_number,
                            allowed_roles=allowed_roles,
                            text=chunk_text,
                            metadata={
                                "doc_id": doc_id,
                                "title": title,
                                "page": page.page_number,
                                "allowed_roles": allowed_roles,
                            },
                        )
                    )
                    global_chunk_counter += 1

        return all_chunks

    def _split_by_structure(self, text: str) -> list[str]:
        """Split page text on headings and clause boundaries."""
        matches = list(re.finditer(self.STRUCTURE_PATTERN, text))
        if not matches:
            return [text]

        sections: list[str] = []
        last_idx = 0
        for m in matches:
            start = m.start()
            if start > last_idx:
                sections.append(text[last_idx:start].strip())
            last_idx = start

        if last_idx < len(text):
            sections.append(text[last_idx:].strip())

        return [s for s in sections if s]

    def _split_by_token_budget(self, text: str) -> list[str]:
        """Split section text into token budget chunks with token overlap."""
        words = text.split()
        # Rough token approximation (1 word ~ 1.33 tokens)
        words_per_chunk = max(1, int(self.max_tokens / 1.33))
        overlap_words = int(self.token_overlap / 1.33)

        if len(words) <= words_per_chunk:
            return [text]

        chunks: list[str] = []
        start = 0
        total_words = len(words)

        while start < total_words:
            end = min(total_words, start + words_per_chunk)
            chunk_words = words[start:end]
            chunks.append(" ".join(chunk_words))

            if end == total_words:
                break
            start += max(1, words_per_chunk - overlap_words)

        return chunks
