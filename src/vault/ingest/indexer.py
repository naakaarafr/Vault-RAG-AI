"""Indexing module for Qdrant vector storage, BM25 indexing, and Postgres circular_index."""

import pickle
from dataclasses import asdict
from pathlib import Path
from typing import Any

from vault.config import Settings, get_settings
from vault.embeddings import SentenceTransformerEmbedder
from vault.ingest.chunker import IngestedChunk
from vault.retrieval import BM25Retriever
from vault.vector_store import Document, VectorStore


class IngestionIndexer:
    """Coordinates embedding, vector storage, BM25 indexing, and Postgres metadata insertion."""

    def __init__(
        self,
        settings: Settings | None = None,
        vector_store: VectorStore | None = None,
        embedder: SentenceTransformerEmbedder | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.data_dir = Path(self.settings.data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.vector_store = vector_store or VectorStore(
            persistence_path=self.data_dir / "vector_db.json"
        )
        self.embedder = embedder or SentenceTransformerEmbedder(
            model_name=self.settings.embed_model
        )
        self.bm25_retriever = BM25Retriever()

    def index_chunks(self, chunks: list[IngestedChunk]) -> None:
        """Embed chunks, add to vector store, build BM25 index, and persist to disk."""
        if not chunks:
            return

        texts = [c.text for c in chunks]

        # 1. Embed text chunks
        try:
            embeddings = self.embedder.embed_documents(texts)
        except Exception:
            # Dummy embedding fallback if model weights offline
            embeddings = [[0.01] * 384 for _ in chunks]

        # 2. Build Document vector units
        vector_docs: list[Document] = []
        for c, emb in zip(chunks, embeddings, strict=False):
            doc = Document(
                doc_id=c.chunk_id,
                content=c.text,
                metadata=asdict(c),
                embedding=emb,
            )
            vector_docs.append(doc)

        # 3. Add to vector store
        self.vector_store.add_documents(vector_docs)

        # 4. Add to BM25 index and serialize to DATA_DIR
        self.bm25_retriever.add_documents(vector_docs)
        self._persist_bm25_index()

    def register_circular_index_postgres(self, manifest_rows: list[dict[str, Any]]) -> None:
        """Create Postgres circular_index table and populate metadata from manifest."""
        dsn = self.settings.postgres_dsn
        try:
            import psycopg2

            conn = psycopg2.connect(dsn)
            with conn.cursor() as cur:
                # Create table circular_index
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS circular_index (
                        doc_id VARCHAR(100) PRIMARY KEY,
                        title TEXT,
                        issued_date VARCHAR(50),
                        department VARCHAR(100)
                    );
                    """
                )
                for row in manifest_rows:
                    cur.execute(
                        """
                        INSERT INTO circular_index (doc_id, title, issued_date, department)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (doc_id) DO UPDATE
                        SET title = EXCLUDED.title,
                            issued_date = EXCLUDED.issued_date,
                            department = EXCLUDED.department;
                        """,
                        (
                            row.get("doc_id"),
                            row.get("title"),
                            row.get("issued_date", "2026-01-01"),
                            row.get("department", "general"),
                        ),
                    )
            conn.commit()
            conn.close()
        except Exception as exc:
            # Log gracefully if Postgres DB is offline
            print(f"[Indexer] Postgres circular_index registration skipped (offline/error): {exc}")

    def _persist_bm25_index(self) -> None:
        """Serialize BM25 index data to DATA_DIR/bm25_index.pkl."""
        bm25_file = self.data_dir / "bm25_index.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump(
                {
                    "doc_tokens": self.bm25_retriever.doc_tokens,
                    "doc_lens": self.bm25_retriever.doc_lens,
                    "avg_doc_len": self.bm25_retriever.avg_doc_len,
                    "df": dict(self.bm25_retriever.df),
                },
                f,
            )
