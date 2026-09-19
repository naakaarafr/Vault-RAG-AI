"""BM25 sparse keyword retriever with RBAC payload filtering."""

import math
import pickle
import re
from collections import Counter
from pathlib import Path

from vault.config import get_settings
from vault.retrieve.protocol import Hit
from vault.vector_store import Document, VectorStore


class BM25Retriever:
    """Okapi BM25 sparse keyword retriever enforcing Role-Based Access Control (RBAC)."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.documents: dict[str, Document] = {}
        self.doc_tokens: dict[str, list[str]] = {}
        self.doc_lens: dict[str, int] = {}
        self.avg_doc_len: float = 0.0
        self.df: Counter[str] = Counter()

        # Attempt auto-load from DATA_DIR if index exists
        self._try_load_index()

    def add_documents(self, docs: list[Document]) -> None:
        """Tokenize and index documents."""
        for doc in docs:
            self.documents[doc.doc_id] = doc
            tokens = self._tokenize(doc.content)
            self.doc_tokens[doc.doc_id] = tokens
            self.doc_lens[doc.doc_id] = len(tokens)
            for token in set(tokens):
                self.df[token] += 1

        total_words = sum(self.doc_lens.values())
        num_docs = len(self.documents)
        self.avg_doc_len = (total_words / num_docs) if num_docs > 0 else 0.0

    def search(
        self,
        query: str,
        k: int = 5,
        roles: list[str] | None = None,
    ) -> list[Hit]:
        """Perform BM25 keyword search with RBAC payload filtering."""
        if not self.documents or not query.strip():
            return []

        query_tokens = self._tokenize(query)
        num_docs = len(self.documents)
        scores: dict[str, float] = {}
        user_roles = set(roles) if roles else None

        for doc_id, tokens in self.doc_tokens.items():
            doc = self.documents[doc_id]
            allowed_roles = doc.metadata.get("allowed_roles", ["public"])
            if isinstance(allowed_roles, str):
                allowed_roles = [r.strip() for r in allowed_roles.split(";")]

            # RBAC Filter: Caller roles must intersect with chunk allowed_roles
            if user_roles is not None and not user_roles.intersection(set(allowed_roles)):
                continue

            doc_len = self.doc_lens[doc_id]
            tf_counter = Counter(tokens)
            score = 0.0

            for token in query_tokens:
                if token not in self.df:
                    continue
                tf = tf_counter[token]
                df_val = self.df[token]
                idf = math.log((num_docs - df_val + 0.5) / (df_val + 0.5) + 1.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / self.avg_doc_len))
                score += idf * (numerator / denominator)

            if score > 0:
                scores[doc_id] = score

        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]

        hits: list[Hit] = []
        for doc_id, score in sorted_docs:
            doc = self.documents[doc_id]
            hits.append(
                Hit(
                    chunk_id=doc.doc_id,
                    doc_id=doc.metadata.get("doc_id", doc.doc_id),
                    page=doc.metadata.get("page", 1),
                    text=doc.content,
                    score=score,
                    metadata=doc.metadata,
                )
            )

        return hits

    def _try_load_index(self) -> None:
        """Load serialized BM25 index from DATA_DIR/bm25_index.pkl if present."""
        data_dir = Path(get_settings().data_dir)
        index_file = data_dir / "bm25_index.pkl"
        db_file = data_dir / "vector_db.json"

        if db_file.exists():
            store = VectorStore(persistence_path=db_file)
            self.documents = store.documents

        if index_file.exists():
            try:
                with open(index_file, "rb") as f:
                    data = pickle.load(f)
                    self.doc_tokens = data.get("doc_tokens", {})
                    self.doc_lens = data.get("doc_lens", {})
                    self.avg_doc_len = data.get("avg_doc_len", 0.0)
                    self.df = Counter(data.get("df", {}))
            except Exception:
                pass

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())
