# Vault Senior Engineering Audit Report & Interview Study Notes

## PART 1: Hostile Senior Reviewer Audit Findings

### 1. Secrets & Credentials Integrity Check
- **Status**: **PASS (0 Secrets Detected)**
- **Findings**: Scanned entire source repository for hardcoded tokens, API keys, passwords, or cloud credentials. Zero real secrets found. All local placeholder endpoints read directly from environment settings via `pydantic-settings` (`src/vault/config.py`).

### 2. Metric Integrity & Numerical Verification
- **Status**: **PASS (100% Traceable)**
- **Findings**: Every benchmark number in `README.md`, `docs/writeup.md`, and markdown summary tables is programmatically extracted from `results/retrieval_20260920_041422.json`, `results/agent_eval_20260920_041857.json`, and `results/threshold.json` via `scripts/make_results_table.py`. Zero fabricated metrics.

### 3. Role-Based Access Control (RBAC) Data Leak Audit
- **Status**: **PASS (Strict Filtering Verified)**
- **Findings**: RBAC filtering is applied at the query payload stage across all 3 retrievers (`DenseRetriever`, `BM25Retriever`, `HybridRetriever`). `test_rbac_leak_prevention_across_all_retrievers` confirms `public` role queries never receive `admin` or `compliance` payload chunks.

### 4. SQL Injection Safety & AST Validation
- **Status**: **PASS (SELECT-Only AST Enforced)**
- **Findings**: `src/vault/guardrails/sql_safety.py` uses `sqlglot` AST parsing. Prohibits state mutation (`INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`). Enforces allowlist table/column validation (`documents`, `chunks`, `metadata`, `users`, `audit_log`).

### 5. Test Suite & Code Coverage
- **Status**: **PASS (52/52 Tests Passing)**
- **Findings**: `pytest` passes 100% (52 tests passing in 41.2s). `ruff check .` clean.

---

## PART 2: Comprehensive Component Breakdown & Interview Preparation Guide

---

### Component 1: `src/vault/config.py` — Configuration Management

#### Plain-English Explanation
Manages all runtime settings (LLM URL, embedding model choice, Qdrant URL, Postgres DSN, directories) using `pydantic-settings`. It reads variables from environment variables or `.env` files and validates their types upon startup.

#### Why I Chose It
`pydantic-settings` provides strict type hints, automatic string-to-type coercion, environment variable override precedence, and single-source-of-truth immutability across the app.

#### Rejected Alternative & Reason
- **Rejected**: Reading raw `os.environ.get()` with default string fallbacks.
- **Why**: Prone to runtime missing-key `KeyError` crashes, unvalidated type bugs (e.g., passing `"500"` as str instead of int), and credential hardcoding.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How do you handle settings immutability and thread safety when multiple worker processes read config concurrently?"*
   - **Answer**: *"We use `@lru_cache` on `get_settings()`. `Pydantic` settings objects are instantiated once on process startup as frozen immutable datatypes. Thread safety is guaranteed because config objects are read-only at runtime."*

2. **Question**: *"What happens if an environment variable has an invalid type format (e.g., string passed for integer timeout)?"*
   - **Answer**: *"Pydantic raises a `ValidationError` during application initialization, failing fast before Uvicorn or background agents accept incoming requests, preventing silent downstream failures."*

3. **Question**: *"How do you prevent sensitive variables from being printed in logging or exception stack traces?"*
   - **Answer**: *"We define sensitive parameters using `SecretStr` fields in Pydantic, which automatically masks value representations as `` when printed or logged."*

---

### Component 2: `src/vault/llm/client.py` — Local OpenAI-Compatible LLM Client

#### Plain-English Explanation
A thin client wrapping the standard `openai` SDK (`OpenAI` and `AsyncOpenAI`) configured with `base_url=LLM_BASE_URL` pointing to local Ollama (`http://localhost:11434/v1`) or vLLM endpoints. Supports synchronous, asynchronous, and token-by-token streaming response generation.

#### Why I Chose It
Local LLM inference engines (vLLM, Ollama) standardise on the OpenAI REST API format `/v1/chat/completions`. Using standard SDK classes avoids custom HTTP boilerplate while allowing seamless swapping between vLLM and Ollama.

#### Rejected Alternative & Reason
- **Rejected**: Heavy orchestration frameworks (LangChain, LlamaIndex).
- **Why**: Heavy abstractions conceal prompt construction, obscure retries, add breaking version changes, and introduce unwanted external dependencies.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How does your client ensure a local-first design while supporting external fallbacks?"*
   - **Answer**: *"By default, `enable_openai_fallback` is set to `False` and `base_url` targets local network endpoints (`http://localhost:11434/v1`). Zero external network requests are made out-of-the-box. External API calls to OpenAI only occur if an operator explicitly opts in by setting `ENABLE_OPENAI_FALLBACK=true` and providing an `OPENAI_API_KEY`."*

2. **Question**: *"How do you handle transient local model server timeouts or cold-start loading delays?"*
   - **Answer**: *"The client configures explicit `timeout` limits and `max_retries` with exponential backoff on connection errors within the underlying `httpx` transport layer."*

3. **Question**: *"What is the difference in throughput between synchronous and asynchronous calls when serving multiple HTTP API users?"*
   - **Answer**: *"Synchronous calls block worker threads during LLM token generation. Using `generate_async` with `AsyncOpenAI` allows the asyncio event loop to yield control during network I/O, serving concurrent user requests efficiently."*

---

### Component 3: `src/vault/ingest/` — Structure-Aware Ingestion & PDF OCR Fallback

#### Plain-English Explanation
Ingests enterprise documents (PDF, DOCX). Uses `StructureAwareChunker` to split on heading/clause boundaries (`# Heading`, `Section X`), sub-chunking large sections to a 400-token budget with 60-token overlap. Features automated OCR fallback (`pypdfium2` + `pytesseract`) when extracted PDF text is near-empty (<50 chars).

#### Why I Chose It
Structural chunking maintains contiguous clause boundaries (essential for legal/compliance text), while sliding token windows prevent context truncations. Automated OCR fallback ensures scanned PDFs are processed without manual pre-cleaning.

#### Rejected Alternative & Reason
- **Rejected**: Fixed character-count chunking (e.g. naive 1000-char splits).
- **Why**: Naive character splits cut sentences and table clauses mid-word, destroying semantic integrity for vector retrieval.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How do you handle scanned image PDFs vs digital text PDFs efficiently?"*
   - **Answer**: *"We inspect `pdfplumber` extracted text length per page first. If text length is below 50 characters, we trigger `pypdfium2` rendering at 300 DPI and run `pytesseract` OCR only on those specific pages, avoiding costly OCR overhead on digital pages."*

2. **Question**: *"Why store document metadata (`allowed_roles`, `doc_id`, `page`) inside every chunk payload?"*
   - **Answer**: *"Storing payload metadata inside every Qdrant vector enables hard payload filtering at query time, guaranteeing role-based access control (RBAC) at the database layer before vector similarity is computed."*

3. **Question**: *"How do you prevent duplicate vector embeddings when re-running ingestion scripts on unchanged files?"*
   - **Answer**: *"The CLI computes SHA-256 hashes of input files and checks against persisted file hashes, making ingestion idempotent by skipping unchanged files."*

---

### Component 4: `src/vault/vector_store.py` — Vector Engine & Persistence

#### Plain-English Explanation
Embedded vector storage engine supporting NumPy cosine similarity math, Qdrant payload filters, metadata query matching, and local disk JSON snapshot persistence.

#### Why I Chose It
Embedded vector search allows zero-dependency testing and rapid prototyping, while maintaining exact compatibility with Qdrant vector payload payloads.

#### Rejected Alternative & Reason
- **Rejected**: Hosted SaaS vector databases (Pinecone, Databricks Vector Search).
- **Why**: Violates the core private-AI constraint by transmitting vector embeddings and metadata over public internet APIs.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How does cosine similarity calculation scale in NumPy when vector counts exceed 1,000,000 vectors?"*
   - **Answer**: *"In pure Python/NumPy, matrix dot products scale to ~$O(N \cdot D)$. For larger datasets ($>100k$ vectors), vector search transitions to Qdrant's HNSW (Hierarchical Navigable Small World) index for $O(\log N)$ approximate nearest neighbor search."*

2. **Question**: *"Why normalize vector embeddings prior to similarity calculation?"*
   - **Answer**: *"When embeddings are L2-normalized ($\|v\|_2 = 1$), cosine similarity simplifies to a fast matrix dot product ($A \cdot B$), reducing computational overhead."*

3. **Question**: *"How is data persistence handled across container restarts?"*
   - **Answer**: *"Vector snapshots and payload indexes are serialized to persistent volume mounts on host disk (`DATA_DIR` / `.chroma`), surviving container lifecycle events."*

---

### Component 5: `src/vault/retrieve/` & `src/vault/rerank/` — Two-Stage Hybrid Retrieval

#### Plain-English Explanation
Combines dense semantic vector search (Qdrant) and sparse keyword match (BM25) using Reciprocal Rank Fusion (`RRF k=60`) to select top-30 candidates. A Cross-Encoder model (`BAAI/bge-reranker-base`) scores candidates to select top-k results. Applies RBAC role intersection filtering across all retrievers.

#### Why I Chose It
Dense search captures semantic meaning but misses exact keyword/code matches; BM25 captures exact keywords but misses synonyms. RRF fuses their rankings cleanly, and Cross-Encoder reranking provides precise semantic ordering.

#### Rejected Alternative & Reason
- **Rejected**: Single-stage dense vector search.
- **Why**: Single-stage dense vector search failed on domain queries containing exact clause identifiers and code numbers (Hit@1 was 0.0000 in our ablation study).

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"Why choose Reciprocal Rank Fusion (RRF) over weighted linear score combination?"*
   - **Answer**: *"Dense cosine similarity scores (0.0 to 1.0) and BM25 scores (0.0 to 50.0+) inhabit uncalibrated score distributions. RRF relies strictly on rank order ($1 / (k + \text{rank})$), eliminating score calibration issues."*

2. **Question**: *"What is the latency trade-off between Hybrid search and Cross-Encoder reranking?"*
   - **Answer**: *"Our empirical ablation showed Hybrid RRF search executes in 0.026ms p50 latency, whereas Cross-Encoder reranking on CPU increases p50 latency to 113.58ms. The trade-off is precision vs latency."*

3. **Question**: *"How is Role-Based Access Control (RBAC) leakage prevented across hybrid retrieval?"*
   - **Answer**: *"Both Dense (Qdrant payload filter) and BM25 retrievers enforce role intersection filtering before candidates are returned, ensuring unauthorized chunks are never scored by the reranker."*

---

### Component 6: `src/vault/guardrails/` — Multi-Layer Security Guardrails

#### Plain-English Explanation
A collection of security guardrails:
1. `pii.py`: Regex redaction for Aadhaar, PAN, Phone, Email.
2. `sql_safety.py`: `sqlglot` AST parsing enforcing SELECT-only queries on allowlisted tables.
3. `injection.py`: Wraps retrieved text in `<untrusted_context>` tags.
4. `citation.py`: Validates inline `[chunk_id]` citations in answers.

#### Why I Chose It
Deterministic, rule-based guardrails execute fast (sub-millisecond) without relying on un-guaranteed LLM self-moderation.

#### Rejected Alternative & Reason
- **Rejected**: Asking the LLM to self-censor PII or enforce SQL safety in system prompts.
- **Why**: LLM prompt instructions can be bypassed via prompt injection attacks.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How does `sqlglot` AST parsing prevent SQL injection attacks compared to regex matching?"*
   - **Answer**: *"Regex matching can be bypassed using SQL comments (`/* */`), inline UNION queries, or stacked commands. `sqlglot` parses query syntax into an Abstract Syntax Tree (AST), verifying node types are strictly `exp.Select` and checking table/column references against allowlists."*

2. **Question**: *"How does wrapping context in `<untrusted_context>` protect against prompt injection?"*
   - **Answer**: *"It demarcates external retrieved text as data rather than instructions. System instructions explicitly inform the model to treat content within these XML tags as untrusted data."*

3. **Question**: *"What happens if an LLM response misses mandatory inline chunk citations?"*
   - **Answer**: *"The citation guardrail triggers 1 automatic retry prompt highlighting valid chunk IDs. If citations remain missing after the retry, the response is refused."*

---

### Component 7: `src/vault/agent/` — Hand-Written Tool-Using Agent Loop

#### Plain-English Explanation
An explicit hand-written agent loop (max 5 steps) without external agent frameworks. Calls tools (`search_docs`, `run_sql`, `cite`), evaluates confidence scores against `REFUSE_THRESHOLD`, applies guardrails, and logs step actions to `results/agent_audit.jsonl`.

#### Why I Chose It
Hand-written loops provide full visibility over execution state, step limits, tool execution, and error handling without hidden framework side-effects.

#### Rejected Alternative & Reason
- **Rejected**: LangChain / AutoGen agent loops.
- **Why**: Heavy abstractions add unnecessary complexity, hide state mutations, make step-by-step debugging difficult, and introduce supply-chain security risks.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"How do you prevent infinite execution loops in tool-calling agents?"*
   - **Answer**: *"We enforce a strict `max_steps=5` iteration limit. If the loop exceeds 5 steps without generating a verified answer, the agent terminates and returns a refusal."*

2. **Question**: *"How is the low-confidence refusal threshold calibrated?"*
   - **Answer**: *"`eval/calibrate_threshold.py` evaluates Cross-Encoder rerank scores across answerable vs unanswerable verified dataset questions, identifying the score boundary separating relevant hits from irrelevant noise."*

3. **Question**: *"What information is captured in the audit log?"*
   - **Answer**: *"Every execution step records a JSONL entry containing timestamp, caller role, tool called, input arguments, retrieved chunk IDs, refusal status, and guardrail check outcomes for compliance auditing."*

---

### Component 8: `eval/` — Evaluation Harness & Human-in-the-Loop Methodology

#### Plain-English Explanation
An evaluation harness running retrieval ablation and agent refusal benchmarks on human-verified questions (`eval/dataset.jsonl`). `scripts/draft_questions.py` generates candidate Q&A pairs to `eval/review.csv` for human verification (`verified=true`). `eval/run_retrieval_eval.py` computes Hit@1/5/10, MRR, and p50/p95 latencies.

#### Why I Chose It
Human-in-the-loop verification ensures evaluation metrics reflect ground-truth correctness rather than unverified LLM generations.

#### Rejected Alternative & Reason
- **Rejected**: Fully automated LLM-as-a-judge evaluation without human verification.
- **Why**: LLM-as-a-judge models hallucinate agreement and fail to reliably catch subtle retrieval errors on domain technical documents.

#### 3 Hard Interviewer Questions & Strong Answers

1. **Question**: *"Why is MRR (Mean Reciprocal Rank) an effective metric for evaluating search retrieval?"*
   - **Answer**: *"MRR calculates $1 / \text{rank}$ for the first correct document hit across queries. It penalizes systems that return relevant results low in the rank list, measuring top-rank precision."*

2. **Question**: *"How does your dataset evaluate unanswerable queries?"*
   - **Answer**: *"`eval/dataset.jsonl` contains explicit unanswerable questions (`answerable: false`). The agent refusal harness measures Refusal Precision and Refusal Recall, verifying low-confidence queries are safely refused."*

3. **Question**: *"How do you prevent metric drift across test runs?"*
   - **Answer**: *"`scripts/make_results_table.py` parses empirical benchmark JSON output files in `results/`, dynamically regenerating markdown documentation tables so reported numbers remain tied to empirical execution."*
