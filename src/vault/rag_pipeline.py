"""Explicit Retrieval-Augmented Generation (RAG) pipeline."""

from dataclasses import dataclass, field
from typing import Any

from vault.llm import LLMClient
from vault.retrieval import HybridResult, HybridRetriever


@dataclass
class SourceCitation:
    """Explicit source attribution for retrieved context."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    relevance_score: float = 0.0


@dataclass
class RAGResult:
    """Structured RAG response container."""

    query: str
    answer: str
    sources: list[SourceCitation]
    raw_context: str


class RAGPipeline:
    """RAG pipeline coordinating hybrid retrieval, prompt formatting, and local LLM execution."""

    SYSTEM_PROMPT = (
        "You are Vault, an enterprise local AI assistant. "
        "Answer the user's question accurately using ONLY the provided reference context. "
        "If the answer cannot be determined from the provided context, state clearly that "
        "the information is not available in the provided documents. "
        "Do not invent or assume any facts outside the context."
    )

    def __init__(
        self,
        retriever: HybridRetriever,
        llm_client: LLMClient | None = None,
        top_k: int = 3,
    ) -> None:
        self.retriever = retriever
        self.llm_client = llm_client or LLMClient()
        self.top_k = top_k

    def format_context(self, results: list[HybridResult]) -> str:
        """Format retrieved search results into a clean context text block."""
        if not results:
            return "No relevant context documents found."

        context_blocks = []
        for idx, res in enumerate(results, start=1):
            source_label = res.metadata.get("doc_id", res.doc_id)
            block = f"[Source {idx}: {source_label}]\n{res.content}"
            context_blocks.append(block)

        return "\n\n".join(context_blocks)

    def query(
        self,
        question: str,
        filter_metadata: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> RAGResult:
        """Execute synchronous end-to-end RAG workflow."""
        # 1. Retrieve hybrid context
        retrieved_items = self.retriever.search(
            query=question,
            top_k=self.top_k,
            filter_metadata=filter_metadata,
        )

        # 2. Format context and source citations
        formatted_context = self.format_context(retrieved_items)
        sources = [
            SourceCitation(
                doc_id=item.doc_id,
                content=item.content,
                metadata=item.metadata,
                relevance_score=item.score,
            )
            for item in retrieved_items
        ]

        # 3. Construct OpenAI messages payload
        user_prompt = (
            f"Context Documents:\n{formatted_context}\n\n"
            f"User Question: {question}\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        # 4. Generate answer from local LLM
        answer = self.llm_client.generate(messages=messages, temperature=temperature)

        return RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            raw_context=formatted_context,
        )

    async def query_async(
        self,
        question: str,
        filter_metadata: dict[str, Any] | None = None,
        temperature: float = 0.1,
    ) -> RAGResult:
        """Execute asynchronous end-to-end RAG workflow."""
        retrieved_items = self.retriever.search(
            query=question,
            top_k=self.top_k,
            filter_metadata=filter_metadata,
        )

        formatted_context = self.format_context(retrieved_items)
        sources = [
            SourceCitation(
                doc_id=item.doc_id,
                content=item.content,
                metadata=item.metadata,
                relevance_score=item.score,
            )
            for item in retrieved_items
        ]

        user_prompt = (
            f"Context Documents:\n{formatted_context}\n\n"
            f"User Question: {question}\n\n"
            "Answer:"
        )

        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        answer = await self.llm_client.generate_async(messages=messages, temperature=temperature)

        return RAGResult(
            query=question,
            answer=answer,
            sources=sources,
            raw_context=formatted_context,
        )
