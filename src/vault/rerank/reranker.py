"""Cross-Encoder reranker module using sentence-transformers."""

from vault.config import Settings, get_settings
from vault.retrieve.protocol import Hit


class CrossEncoderReranker:
    """Reranks candidate hits using a Cross-Encoder model (BAAI/bge-reranker-base)."""

    def __init__(
        self,
        model_name: str | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.model_name = model_name or self.settings.rerank_model
        self._model = None

    @property
    def model(self):
        """Lazy loader for CrossEncoder model."""
        if self._model is None:
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
            except Exception:
                # Graceful fallback scorer if weights or model unavailable offline
                self._model = "fallback"
        return self._model

    def rerank(self, query: str, hits: list[Hit], top_k: int = 5) -> list[Hit]:
        """Rerank candidate hits using CrossEncoder score."""
        if not hits or not query.strip():
            return []

        pairs = [[query, h.text] for h in hits]

        if self.model != "fallback":
            try:
                scores = self.model.predict(pairs)
            except Exception:
                scores = [h.score for h in hits]
        else:
            # Deterministic fallback score simulation based on term overlap & base score
            query_terms = set(query.lower().split())
            scores = []
            for h in hits:
                text_terms = set(h.text.lower().split())
                overlap = len(query_terms.intersection(text_terms))
                scores.append(h.score + (overlap * 0.1))

        # Attach reranker scores and re-sort descending
        reranked_hits: list[Hit] = []
        for hit, score in zip(hits, scores, strict=False):
            reranked_hits.append(
                Hit(
                    chunk_id=hit.chunk_id,
                    doc_id=hit.doc_id,
                    page=hit.page,
                    text=hit.text,
                    score=float(score),
                    metadata=hit.metadata,
                )
            )

        reranked_hits.sort(key=lambda h: h.score, reverse=True)
        return reranked_hits[:top_k]
