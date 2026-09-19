"""Retriever protocol and standard Hit data contract."""

from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass
class Hit:
    """Standard search result hit returned by all retrievers."""

    chunk_id: str
    doc_id: str
    page: int
    text: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Retriever(Protocol):
    """Protocol interface for all Vault retrieval implementations."""

    def search(
        self,
        query: str,
        k: int = 5,
        roles: list[str] | None = None,
    ) -> list[Hit]:
        """Search query returning top-k Hit objects filtered by caller roles."""
        ...
