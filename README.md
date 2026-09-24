# Vault: Self-Hosted Enterprise Guardrailed RAG Stack

Vault is an enterprise-grade Retrieval-Augmented Generation (RAG) platform built with a local-first architecture for private AI deployments. Built with Python 3.11, explicit modular architecture (no LangChain/LlamaIndex), role-based access control (RBAC), multi-stage hybrid retrieval with Cross-Encoder reranking, and hand-written agent guardrails (PII redaction, prompt injection defense, SQL AST safety via `sqlglot`, and low-confidence refusals), Vault operates with no external calls by default and provides an opt-in OpenAI fallback model for high availability.

## Architecture

```mermaid
graph TD
    Client[Web UI / REST API Client] -->|POST /ask| API[FastAPI Web Server]
    API -->|Question + Role| Agent[VaultAgent Loop]
    
    subgraph Guardrails Layer
        P1[Input Sanitization]
        P2[PII Redaction Engine]
        P3[SQL AST Safety Check]
        P4[Untrusted Context Wrapper]
        P5[Citation Verifier]
    end

    subgraph Retrieval Pipeline
        Dense[Dense Vector Index]
        BM25[BM25 Sparse Index]
        RRF[Reciprocal Rank Fusion]
        Reranker[BAAI/bge-reranker-base]
    end

    Agent --> P1
    Agent -->|search_docs| Dense
    Agent -->|search_docs| BM25
    Dense --> RRF
    BM25 --> RRF
    RRF --> Reranker
    Reranker -->|Top-K Hits| P4
    P4 -->|Context| LLM[Local Ollama / vLLM Server]
    LLM -->|On Failure (Opt-in)| Fallback[OpenAI Fallback Model: gpt-4o-mini]
    LLM -->|Response| P5
    Fallback -->|Response| P5
    P5 --> P2
    P2 -->|Final Output| API
```

---

## Quickstart

Spin up the entire stack (Qdrant, Postgres, Ollama server, automatic model downloader, and Vault API) locally:

```bash
docker compose up --build
```

Access the interactive web workspace at **`http://localhost:8000`** or perform health check:

```bash
curl http://localhost:8000/health
```

---

## Local-First & High-Availability Model Support

- **Local-First Default**: All embeddings (`BAAI/bge-small-en-v1.5`) and LLM chat completions run locally against Ollama or vLLM endpoints with zero external calls by default.
- **Opt-in OpenAI Fallback**: If the local LLM server is offline or fails to respond, `LLMClient` can seamlessly fall back to an OpenAI model (e.g., `gpt-4o-mini`). Fallback is disabled (`False`) by default and only activates when `ENABLE_OPENAI_FALLBACK=true` and `OPENAI_API_KEY` is provided.
- **Privacy & Resilience**: Guarantees data privacy by keeping operations local by default while offering high-availability resilience when enabled.

### OpenAI Fallback Configuration (Opt-in)

Set the following environment variables in `.env`:

```env
OPENAI_API_KEY=your-api-key-here
OPENAI_MODEL=gpt-4o-mini
OPENAI_BASE_URL=https://api.openai.com/v1
ENABLE_OPENAI_FALLBACK=true
```


---

## Benchmark & Empirical Evaluation Results

*(Generated directly from empirical runs in `results/` using `python scripts/make_results_table.py`)*

### 1. Retrieval Strategy Ablation

| Configuration | Hit@1 | Hit@5 | Hit@10 | MRR | p50 Latency (ms) | p95 Latency (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `dense` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.001 |
| `hybrid` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.026 | 0.111 |
| `hybrid_rerank` | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 113.586 | 8353.071 |

### 2. Agent Refusal & Guardrail Accuracy

| Refusal / Guardrail Metric | Score |
| :--- | :---: |
| **Refusal Precision** | 0.0000 |
| **Refusal Recall** | 0.0000 |
| **Refusal F1** | 0.0000 |
| **Overall Accuracy** | 0.2000 |

### 3. Operating Cost per 1,000 Queries

| Resource Component | Local Hardware Cost | External API Runtime Cost per 1k Queries |
| :--- | :---: | :---: |
| **LLM Inference** | Local Ollama / vLLM | **$0.00** |
| **Dense Vector Embeddings** | `BAAI/bge-small-en-v1.5` | **$0.00** |
| **Cross-Encoder Reranking** | `BAAI/bge-reranker-base` | **$0.00** |
| **Total Operating Cost** | **Self-Hosted** | **$0.00** |

---

## Guardrail Behaviours & Examples

1. **Mandatory Citations**:
   - *Behaviour*: Answers must cite valid retrieved chunk IDs in format `[chunk_id]`.
   - *Example*: `"Vault enforces RBAC isolation at the vector payload layer [doc_001_p1_c0]."`
2. **Low-Confidence Refusal**:
   - *Behaviour*: If top reranker score is below calibrated `REFUSE_THRESHOLD` (0.002), query is safely refused.
   - *Example*: `"I am sorry, but I do not have sufficient authoritative information..."`
3. **PII Redaction**:
   - *Behaviour*: Regex patterns automatically redact sensitive identifiers.
   - *Example*: `"User phone +91-9876543210"` -> `"User phone [REDACTED_PHONE]"`.
4. **SQL AST Safety**:
   - *Behaviour*: `sqlglot` validates queries to allow `SELECT` statements on allowlisted tables only.
   - *Example*: `DROP TABLE chunks;` -> `SQL Safety Security Violation: Forbidden statement type 'DROP'.`
5. **Prompt Injection Defense**:
   - *Behaviour*: Context wrapped in `<untrusted_context>` tags to isolate model instructions.

---

## Limitations

- **Dataset Scale**: Current evaluation benchmark is conducted on a human-verified 5-question seed dataset (`eval/dataset.jsonl`). Large-scale evaluation across 10,000+ domain docs remains a TODO.
- **CPU Rerank Latency**: Heavy Cross-Encoder reranking (`BAAI/bge-reranker-base`) running on CPU adds high latency (~113ms p50, 8.3s p95) compared to hybrid search alone (0.026ms p50). GPU acceleration or model quantization is recommended for production.
- **Single-Node Persistence**: Qdrant vector storage and SQLite metadata run as single-node instances without distributed sharding.
