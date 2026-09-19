# Architecture Decision Records (ADRs)

## ADR-001: Explicit HTTP Client for Local OpenAI-Compatible LLM Server

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Vault requires local LLM access via an OpenAI-compatible API endpoint (`LLM_BASE_URL` pointing to Ollama or vLLM) with zero hosted API dependencies.
* **Decision**: Implement a thin, explicit wrapper using `httpx.AsyncClient` / `httpx.Client` directly around the standard `/v1/chat/completions` REST specification in `src/vault/llm.py` instead of importing heavy SDKs or framework wrappers (LangChain / LlamaIndex / `openai`).

## ADR-002: Configuration Management via `pydantic-settings`

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Vault requires type-safe, validated environment settings loaded from `.env` files with zero hardcoded credentials or magic parameters in source code.
* **Decision**: Use `pydantic_settings.BaseSettings` in `src/vault/config.py`.

## ADR-003: Embedded Local Vector Database with Cosine Similarity Index

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Enterprise local RAG needs high-speed, zero-external-network vector persistence, exact or approximate similarity search, metadata filtering, and JSON/SQLite array storage.
* **Decision**: Implement an explicit vector index engine in `src/vault/vector_store.py` with normalized vector dot product (cosine similarity), metadata query filtering, and JSON/disk snapshot persistence.

## ADR-004: Sliding-Window Character Chunker with Overlap Protection

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Documents must be chunked into manageable segments for embedding generation while preserving contextual boundary coherence.
* **Decision**: Implement a configurable sliding-window text chunker in `src/vault/ingestion/chunker.py` with boundary snapping to sentence/paragraph breaks.

## ADR-005: Reciprocal Rank Fusion (RRF) for Hybrid Retrieval

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Combining dense semantic vector retrieval with sparse keyword match (BM25) significantly improves search recall and precision for domain technical queries.
* **Decision**: Implement a pure Python BM25 retriever and Reciprocal Rank Fusion (RRF) algorithm in `src/vault/retrieval.py` using formula `RRF_Score(d) = sum(1 / (k + rank_i(d)))`.

## ADR-006: Explicit Source Attribution and Prompt Engineering Contract

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: RAG responses in enterprise environments must provide verifiable source attribution and refuse ungrounded hallucinations.
* **Decision**: Enforce structured `RAGResult` outputs containing answer text and explicit `sources` metadata in `src/vault/rag_pipeline.py`.

## ADR-007: Rule-Based & Pattern-Matching Dual-Tier Guardrails

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Private-AI enterprise deployment requires strict guardrails for PII redaction, prompt injection defense, and output groundedness checking.
* **Decision**: Build explicit `InputGuardrail` and `OutputGuardrail` modules in `src/vault/guardrails/` combining deterministic regex patterns (Email, SSN, Credit Cards, API Keys, system override signatures) and lexical token overlap groundedness metrics.

## ADR-008: Explicit Deterministic Tool-Calling Agent Loop

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Autonomous agents require tool execution capabilities (RAG query, document search, calculation) protected by guardrails.
* **Decision**: Build `VaultAgent` in `src/vault/agent.py` using an explicit dispatch dictionary and multi-step loop with input sanitization, tool selection, and response validation.

## ADR-009: `openai` Python SDK Client Wrapper for vLLM & Ollama Local Standard

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Local LLM runtimes (vLLM, Ollama `/v1`) conform to the OpenAI REST specification.
* **Decision**: Implement `src/vault/llm/client.py` using `openai.OpenAI` and `openai.AsyncOpenAI` with `base_url` pointing to `LLM_BASE_URL`.

## ADR-010: Structure-Aware Two-Stage Document Chunking Strategy

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Enterprise documents contain structural boundaries (headings, clause numbers) that contain contiguous semantic meaning.
* **Decision**: Implement `StructureAwareChunker` in `src/vault/ingest/chunker.py`. First splits text on structural heading regex boundaries (`# Heading`, `Section X`, `1.1 Clause`), then sub-splits oversized blocks by token budget (400 tokens) with 60-token sliding overlap.

## ADR-011: Automated OCR Fallback for Scanned PDF Pages

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Enterprise PDF files frequently contain scanned images or non-selectable text pages.
* **Decision**: Build `PDFLoader` in `src/vault/ingest/loaders.py` with automatic OCR fallback. When page extracted text length is below 50 characters, `pypdfium2` renders the page image at 300 DPI and `pytesseract` extracts OCR text.

## ADR-012: Role-Based Access Control (RBAC) Payload Filtering

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Multi-tenant enterprise systems require strict data isolation based on caller roles (e.g., `public`, `compliance`, `admin`).
* **Decision**: Enforce role intersection filtering across all 3 retrievers (`DenseRetriever`, `BM25Retriever`, `HybridRetriever`). Every query requires caller `roles: list[str]` and rejects any chunk whose `allowed_roles` metadata does not intersect with the caller's active roles.

## ADR-013: Two-Stage Hybrid Retrieval & Cross-Encoder Reranking Architecture

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: First-stage vector search excels at high recall, while deep Cross-Encoder models excel at fine-grained semantic relevance scoring.
* **Decision**: Implement two-stage retrieval in `src/vault/retrieve/` and `src/vault/rerank/`. Stage 1 fuses top-30 dense vector hits and top-30 BM25 sparse hits via Reciprocal Rank Fusion (`RRF k=60`). Stage 2 scores the top-30 candidates with `CrossEncoderReranker` (`BAAI/bge-reranker-base`) and returns top-k.

## ADR-014: Human-in-the-Loop Verified Evaluation Methodology

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Automated benchmark metrics must reflect true human-verified ground-truth correctness rather than unverified LLM generations.
* **Decision**: Implement two-phase dataset workflow: `scripts/draft_questions.py` generates candidate Q&A pairs to `eval/review.csv`. Human reviewers verify correctness and mark `verified=true`. `eval/run_retrieval_eval.py` strictly filters `eval/dataset.jsonl` for `verified == True` rows.
* **Why**:
  1. Prevents noisy or hallucinated LLM candidate questions from skewing precision metrics.
  2. Supports explicit unanswerable questions (`answerable: false`) for robustness evaluation.

## ADR-015: Hand-Written Tool-Using Agent & Security Guardrails Architecture

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: Enterprise local RAG systems require autonomous tool execution without heavy third-party framework abstraction (LangChain/LlamaIndex), protected by multi-layer security guardrails.
* **Decision**: Implement `VaultAgent` in `src/vault/agent/` and security guardrails in `src/vault/guardrails/`:
  1. **Hand-Written Loop**: Max 5-step iteration loop with explicit step auditing to `results/agent_audit.jsonl`.
  2. **Low-Confidence Refusal**: Automatically refuse questions whose top Cross-Encoder rerank score falls below `REFUSE_THRESHOLD` calibrated by `eval/calibrate_threshold.py`.
  3. **Mandatory Citations**: Validate inline `[chunk_id]` citations with 1 retry step before triggering refusal.
  4. **PII Redaction**: Regex scanning for Aadhaar, PAN, phone, and email tokens.
  5. **SQL AST Safety**: `sqlglot` AST parsing enforcing SELECT-only queries against allowlisted tables/columns.
## ADR-016: FastAPI Web API Layer, Glassmorphism Web UI & Docker Compose Stack

* **Date**: 2026-09-20
* **Status**: Accepted
* **Context**: The Vault enterprise platform requires a clean REST interface, interactive user web frontend, and container orchestration with automatic local LLM model pull.
* **Decision**:
  1. **FastAPI Web Service**: Implemented in `src/vault/api/app.py` providing `GET /health` and `POST /ask` endpoints wrapping `VaultAgent` execution.
  2. **Static Glassmorphism UI**: Built responsive HTML/CSS interface in `src/vault/api/static/index.html` featuring dark mode styling, role selection (`public`, `compliance`, `admin`), status badges, and expandable citation pill metadata (`[chunk_id]`).
  3. **Docker Compose Orchestration**: Finalized `docker-compose.yml` defining `qdrant`, `postgres`, `ollama`, `ollama-pull` initialization service, and `api` service.


