"""Hybrid retrieval combining dense vector similarity and sparse BM25 keyword matching with RRF."""

import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any

from vault.embeddings import BaseEmbedder
from vault.vector_store import Document, SearchResult, VectorStore


@dataclass
class HybridResult:
    """Consolidated result returned by hybrid retriever."""

    doc_id: str
    content: str
    metadata: dict[str, Any]
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None


class BM25Retriever:
    """Pure Python implementation of Okapi BM25 sparse keyword retriever."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: dict[str, Document] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_lens: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Counter[str] = Counter()

    def add_documents(self, docs: list[Document]) -> None:
        """Tokenize and index document text for BM25 search."""
        for doc in docs:
            self.documents[doc.doc_id] = doc
            tokens = self._tokenize(doc.content)
            self.doc_tokens[doc.doc_id] = tokens
            self.doc_lens[doc.doc_id] = len(tokens)
            # Unique document frequency count
            for token in set(tokens):
                self.df[token] += 1

        total_words = sum(self.doc_lens.values())
        num_docs = len(self.documents)
        self.avg_doc_len = (total_words / num_docs) if num_docs > 0 else 0.0

    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """Perform BM25 scoring over indexed documents."""
        if not self.documents or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        num_docs = len(self.documents)
        scores: dict[str, float] = {}

        for doc_id, tokens in self.doc_tokens.items():
            doc_len = self.doc_lens[doc_id]
            tf_counter = Counter(tokens)
            score = 0.0

            for token in query_tokens:
                if token not in self.df:
                    continue
                tf = tf_counter[token]
                df_val = self.df[token]
                # Standard Okapi BM25 IDF formula
                idf = math.log((num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
                # Term frequency normalization
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            if score > 0:
                scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        results: list[SearchResult] = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            results.append(
                SearchResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score,
                )
            )

        return results

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Lowercase alphanumeric tokenizer."""
        return re.findall(r"\b\w+\b", text.lower())


class HybridRetriever:
    """Hybrid retriever using Reciprocal Rank Fusion (RRF) between VectorStore and BM25Retriever."""

    def __init__(
        self,
        vector_store: VectorStore,
        embedder: BaseEmbedder,
        bm25_retriever: BM25Retriever | None = None,
        rrf_k: int = 60,
    ) -> None:
        self.vector_store = vector_store
        self.embedder = embedder
        self.bm25_retriever = bm25_retriever or BM25Retriever()
        self.rrf_k = rrf_k
        # Sync initial docs if vector store has any
        if vector_store.documents:
            self.bm25_retriever.add_documents(list(vector_store.documents.values()))

    def add_documents(self, docs: list[Document]) -> None:
        """Add documents to both vector store and BM25 index."""
        self.vector_store.add_documents(docs)
        self.bm25_retriever.add_documents(docs)

    def search(
        self,
        query: str,
        top_k: int = 5,
        filter_metadata: dict[str, Any] | None = None,
    ) -> list[HybridResult]:
        """Execute hybrid search using Reciprocal Rank Fusion."""
        if not query.strip():
            return []

        # 1. Dense vector search
        query_vector = self.embedder.embed_text(query)
        dense_results = self.vector_store.search(
            query_embedding=query_vector,
            top_k=top_k * 2,
            filter_metadata=filter_metadata,
        )

        # 2. Sparse BM25 search
        sparse_results = self.bm25_retriever.search(
            query=query,
            top_k=top_k * 2,
        )

        # 3. Reciprocal Rank Fusion (RRF) calculation
        rrf_scores: dict[str, float] = {}
        dense_ranks: dict[str, int] = {}
        sparse_ranks: dict[str, int] = {}
        doc_map: dict[str, Document] = {}

        for rank, res in enumerate(dense_results, start=1):
            doc_id = res.doc_id
            dense_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if doc_id not in doc_map:
                doc = self.vector_store.get_document(doc_id)
                if doc:
                    doc_map[doc_id] = doc

        for rank, res in enumerate(sparse_results, start=1):
            doc_id = res.doc_id
            sparse_ranks[doc_id] = rank
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (self.rrf_k + rank))
            if doc_id not in doc_map:
                doc = self.vector_store.get_document(doc_id)
                if doc:
                    doc_map[doc_id] = doc

        # Sort combined documents by RRF score descending
        sorted_doc_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]

        final_results: list[HybridResult] = []
        for doc_id, score in sorted_doc_ids:
            doc = doc_map[doc_id]
            final_results.append(
                HybridResult(
                    doc_id=doc.doc_id,
                    content=doc.content,
                    metadata=doc.metadata,
                    score=score,
                    dense_rank=dense_ranks.get(doc_id),
                    sparse_rank=sparse_ranks.get(doc_id),
                )
            )

        return final_results
