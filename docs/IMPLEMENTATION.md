# Mneme: Implementation Specification

> Agent-facing build document for **Claude Code**.
> This is an executable spec: implement phase by phase, sub-phase by sub-phase. Each sub-phase has explicit tasks, file targets, and acceptance criteria. Do not skip ahead. Open one branch and one PR per phase.

---

## 0. How to use this document

You are implementing Mneme, an agentic RAG system over a personal Obsidian vault, self-hosted with vLLM. The repo owner supervises every phase and reviews each PR. Your job is to implement correctly and explain decisions, not to maximize output speed.

**Operating rules:**

1. **One PR per phase.** Branch name `phase-N-slug`. The PR description must state, in prose, the key engineering decision the phase resolves and any trade-offs taken.
2. **Public repo, zero secrets, zero personal data.** This repository is public from the first commit. You initialize it yourself, hygiene first. Never commit secrets, the owner's Obsidian vault or any data derived from it, or hardcoded personal values (paths, hostnames, domains, usernames, IPs). This rule outranks convenience and speed. See §1.2.
3. **Respect GUARDRAIL blocks.** They mark decisions where a fast/naive default is wrong. Stop and confirm with the owner if you are about to violate one.
4. **Honor the acceptance criteria.** A sub-phase is done only when every criterion passes. Write the test before the implementation where a criterion is testable.
5. **Never couple to a concrete LLM engine.** All model calls go through the `LLMClient` abstraction (see §3.2). vLLM and Ollama must be interchangeable via config.
6. **Config is centralized.** No hardcoded URLs, ports, model names, or paths. Behavior config in `config.yaml` (versioned), secrets/machine-specific values in `.env` (gitignored). Loaded through `pydantic-settings`.
7. **One-command, hardware-agnostic deploy.** The whole system runs with `git clone` + `make run` on any machine. GPU is auto-detected; CPU is the fallback. The same compose stack runs identically on a laptop and on the server. See §1.1.
8. **No silent scope creep.** If a task needs something not specified here, surface it in the PR description rather than inventing requirements.
9. **Prose style in docs/comments/PRs: no em-dashes.**

---

## 1. Tech stack (fixed)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.12 | Backend, ingestion, eval |
| Package manager | `uv` | Single workspace, locked |
| API | FastAPI + uvicorn | Async, SSE streaming |
| Vector DB | Qdrant | Dense + sparse in one collection |
| Embeddings | BGE-M3 (`BAAI/bge-m3`) | Dense + sparse from one model, multilingual |
| Reranker | `BAAI/bge-reranker-v2-m3` | Cross-encoder |
| LLM serving | vLLM (GPU profile) / Ollama (CPU profile) | Auto-selected by hardware; both via OpenAI-compatible API |
| Default model | GPU: `Qwen2.5-7B-Instruct-AWQ` · CPU: `qwen2.5:3b` | Per-profile defaults in `config.yaml` |
| Orchestration | LangGraph | State-graph agent (phase 7) |
| Evaluation | RAGAS + custom retrieval metrics | Phase 5 |
| Observability | Langfuse (self-hosted) | Phase 8 |
| Frontend | React + Vite + TypeScript | Phase 3 |
| Data libs | Polars, DuckDB | Where tabular handling helps |
| Containerization | Docker + Docker Compose (profiles) | All services, one stack |
| Exposure | Cloudflare Tunnel | Production overlay only (phase 8) |

GUARDRAIL: Do not substitute any of these without a PR-level justification approved by the owner. In particular do not swap Qdrant for an in-memory store "to move faster" or BGE-M3 for an API embedding model.

### 1.1 Deployment model (hardware-agnostic, one command)

The system must run anywhere with `git clone` + `make run`, choosing GPU or CPU automatically, with an env override. This is implemented from Phase 0, not deferred.

**Mechanism: Compose profiles + a detection script + env override.**

- `docker-compose.yml` defines always-on services (`qdrant`, `api`, `frontend`) plus two mutually exclusive serving services:
  - `vllm` under profile `gpu`, reserving the NVIDIA device via `deploy.resources.reservations.devices`.
  - `ollama` under profile `cpu`.
- `make run` calls `scripts/detect.sh`, which:
  1. If `SERVING_PROFILE` is set in `.env`, use it verbatim (override wins).
  2. Else probe whether Docker can use an NVIDIA GPU. The check must verify the Docker NVIDIA runtime is actually available (for example `docker info` exposing the `nvidia` runtime, or a throwaway `docker run --rm --gpus all` probe), not merely that `nvidia-smi` exists on the host.
  3. Resolve profile to `gpu` or `cpu`, and from `config.yaml` resolve the matching `LLM_BASE_URL` and `LLM_MODEL` (`http://vllm:8000/v1` vs `http://ollama:11434/v1`).
  4. Export those and run `docker compose --profile <profile> up -d`.
- A production overlay (`docker-compose.prod.yml`) adds Langfuse, auth, and the Cloudflare Tunnel. Introduced in Phase 8. The base stack stays runnable standalone for development.

Behavior requirements:
- On a machine with no usable GPU, `make run` must still succeed end to end on CPU. Never hard-fail because GPU is absent.
- First run downloads the model and the embed/rerank weights into named volumes; document the expected wait.
- `EMBED_DEVICE` and `RERANK_DEVICE` accept `auto|cpu|cuda`; `auto` resolves to `cuda` only when a GPU is visible to the container, else `cpu`.

GUARDRAIL: vLLM CPU mode is not acceptable as the CPU path. CPU path is Ollama with a small model. Do not attempt to run vLLM without a GPU.

### 1.2 Public repository hygiene

The repo is public from commit zero. The cost of a leak is permanent: a secret or a private note pushed once stays in git history forever. Treat this as a hard constraint, not a cleanup step.

**Never committed (enforce via `.gitignore` from the first commit):**
- `.env` and any real secret, token, key, or credential (LLM keys, `AUTH_TOKEN`, Langfuse keys, Cloudflare Tunnel credentials).
- The owner's Obsidian vault and anything derived from it: parsed documents, chunks, embeddings, the Qdrant data volume, model/weight caches.
- The real evaluation golden set (it contains the owner's personal knowledge). See §4 Phase 5.1 for the split.
- Langfuse data and any captured traces (they contain real queries).
- Local build artifacts, `__pycache__`, `node_modules`, `.venv`, coverage, logs.

**Never hardcoded anywhere in tracked files:**
- Vault path, hostnames, domains, usernames, IPs, ports tied to a specific machine. All of these come from `.env` / `config.yaml`. Tracked files use placeholders only.

**Shipped, but safe:**
- `.env.example` with placeholder values only, never real ones.
- Synthetic test fixtures (a tiny fake vault), never real notes.
- A small synthetic golden sample for reproducibility (see Phase 5.1).
- A `LICENSE` and a `README` written for a stranger: generic setup instructions, no personal infrastructure details.

**Tooling (wired in Phase 0.1, before any app code):**
- A comprehensive `.gitignore` is part of the very first commit.
- Secret scanning in `pre-commit` (gitleaks or detect-secrets) so a secret cannot be committed even by accident.
- A `detect-private-key` and large-file guard in the same pre-commit config.

GUARDRAIL: Before the first `git push`, and before any commit that adds data or config, verify no secret or personal data is staged. If secret scanning is not yet wired, wire it before committing anything else. If you are ever unsure whether a file is safe to commit, do not commit it and flag it in the PR.

---

## 2. Repository structure

```
mneme/
├── pyproject.toml            # uv workspace
├── uv.lock
├── .env.example
├── config.yaml               # versioned behavior config (per-profile defaults, chunking, weights)
├── justfile                  # task runner (or Makefile)
├── docker-compose.yml        # base stack: qdrant + profiled serving + api + frontend
├── docker-compose.prod.yml   # production overlay: langfuse, auth, tunnel (phase 8)
├── scripts/
│   └── detect.sh             # GPU detection -> profile + endpoints
├── README.md
├── docs/
│   └── IMPLEMENTATION.md     # this file (the only spec the agent needs)
├── backend/
│   └── mneme/
│       ├── config.py         # pydantic-settings
│       ├── llm/              # LLMClient abstraction + impls
│       ├── retrieval/        # dense, sparse, hybrid, rerank, graph
│       ├── rag/              # prompt assembly, answer synthesis
│       ├── agent/            # LangGraph graph (phase 7)
│       ├── api/              # FastAPI routers
│       └── obs/              # Langfuse instrumentation (phase 8)
├── ingestion/
│   └── mneme_ingest/
│       ├── parser.py         # Obsidian markdown parser
│       ├── chunker.py        # structural chunker
│       ├── graph.py          # wikilink graph extraction
│       ├── embed.py          # BGE-M3 embedding
│       └── index.py          # CLI entrypoint
├── eval/
│   ├── golden/               # curated golden dataset
│   ├── run_eval.py
│   └── report.py
├── frontend/                 # React + Vite
└── tests/
```

---

## 3. Cross-cutting contracts

### 3.1 Configuration

Two layers, loaded into one `Settings` object (pydantic-settings):

- **`config.yaml`** (versioned): application behavior. Per-profile model defaults, chunking params (target size, overlap), retrieval `top_k`/`top_n`, graph re-scoring weights (`alpha`/`beta`/`gamma`), rerank settings.
- **`.env`** (gitignored): machine-specific and secret values.

Minimum `.env` keys:

```
SERVING_PROFILE       # gpu | cpu | (empty = auto-detect)
LLM_BASE_URL          # resolved by detect.sh per profile; overridable
LLM_MODEL             # resolved per profile; overridable
LLM_API_KEY           # dummy for local serving
QDRANT_URL
QDRANT_COLLECTION
EMBED_MODEL           # BAAI/bge-m3
EMBED_DEVICE          # auto | cpu | cuda
RERANK_MODEL          # BAAI/bge-reranker-v2-m3
RERANK_DEVICE         # auto | cpu | cuda
VAULT_PATH
LANGFUSE_*            # phase 8
AUTH_TOKEN            # phase 8
```

`config.yaml` is the source of truth for per-profile defaults; `detect.sh` reads it to fill `LLM_BASE_URL`/`LLM_MODEL` when not already set in `.env`.

### 3.2 LLMClient abstraction

```python
class LLMClient(Protocol):
    async def complete(self, messages: list[Message], **opts) -> str: ...
    async def stream(self, messages: list[Message], **opts) -> AsyncIterator[str]: ...
```

One implementation, `OpenAICompatClient`, talks to both vLLM and Ollama via the `/v1/chat/completions` endpoint. Engine selection is config-only. No other module imports an engine SDK directly.

GUARDRAIL: If any module outside `backend/mneme/llm/` imports `openai`, `vllm`, or `ollama` directly, that is a defect. Route through `LLMClient`.

### 3.3 Core data contracts

```python
class Document:
    id: str                 # stable hash of vault-relative path
    rel_path: str
    title: str
    frontmatter: dict
    tags: list[str]
    wikilinks: list[str]    # outgoing link targets (note titles/paths)
    mtime: float

class Chunk:
    id: str                 # f"{document_id}::{ordinal}"
    document_id: str
    rel_path: str
    heading_path: list[str] # e.g. ["Project", "Architecture", "Retrieval"]
    text: str
    tags: list[str]
    ordinal: int
    token_count: int
```

The `heading_path` is mandatory: every chunk must know which heading hierarchy it lives under, and that context must be prependable at retrieval time.

### 3.4 API contracts

```
POST /index           body: {vault_path?, incremental?}  -> {documents, chunks, links, elapsed_s}
POST /query           body: {question, mode?, filters?}  -> QueryResponse
GET  /query/stream    SSE; same input as /query, streams tokens then a final sources event
GET  /health          -> {status, qdrant, llm, embed}

QueryResponse = {
  answer: str,
  sources: [{rel_path, heading_path, snippet, score}],
  mode: "naive" | "hybrid" | "graph" | "agent",
  trace_id?: str        # phase 8
}
```

`mode` must be switchable per request and default to the best available mode for the current phase. Keeping every mode reachable is required for evaluation.

---

## 4. Phases

Legend: each sub-phase lists **Tasks**, **Files**, **Acceptance**. GUARDRAIL blocks are mandatory checks.

---

### Phase 0: Foundations and serving smoke test

**Goal:** reproducible skeleton; public repo initialized with privacy hardening; model answers over the OpenAI-compatible API. No RAG yet.

**0.1 Repo init + privacy hardening (do this first, before any other file)**
- Tasks: `git init`; author a comprehensive `.gitignore` covering everything in §1.2 (`.env`, vault, derived artifacts, model caches, Qdrant volume, Langfuse data, `node_modules`, `.venv`, caches, logs); `.pre-commit-config.yaml` with secret scanning (gitleaks or detect-secrets), `detect-private-key`, and a large-file guard; `.env.example` with placeholders only; a `LICENSE`; a generic `README` skeleton (no personal infra details). Make the first commit hygiene-only.
- Files: `.gitignore`, `.pre-commit-config.yaml`, `.env.example`, `LICENSE`, `README.md`.
- Acceptance: a fresh clone contains no secrets and no personal data; secret scanning runs in pre-commit and passes; `.env` is ignored; `git status` on a populated `.env` shows it untracked.

GUARDRAIL: This sub-phase precedes everything else. Do not create application code, compose files, or config before `.gitignore` and secret scanning are in place. Do not commit anything until §1.2 is satisfied.

**0.2 Tooling**
- Tasks: init `uv` workspace; ruff + black wired into pre-commit; `justfile` with `fmt`, `lint`, `test`, `run`, `chat`; create the directory tree from §2.
- Files: `pyproject.toml`, `justfile`, extend `.pre-commit-config.yaml`.
- Acceptance: `just lint` and `just test` run green on an empty suite; tree matches §2.

**0.3 Compose stack + GPU detection**
- Tasks: `docker-compose.yml` with always-on `qdrant` plus profiled `vllm` (gpu) and `ollama` (cpu) services; `scripts/detect.sh` implementing the §1.1 logic; `config.yaml` with per-profile model defaults; `make run` wired to detect then `compose up`.
- Files: `docker-compose.yml`, `scripts/detect.sh`, `config.yaml`, `justfile`.
- Acceptance: on a GPU host `make run` selects the `gpu` profile and starts vLLM; on a CPU-only host it selects `cpu` and starts Ollama; `SERVING_PROFILE=cpu make run` forces CPU on a GPU host. Qdrant is healthy in all cases.

GUARDRAIL: CPU path must succeed with no GPU present. Do not let a missing GPU hard-fail `make run`. Do not run vLLM on CPU.

**0.4 Serving + LLMClient**
- Tasks: implement `OpenAICompatClient` talking to whichever engine the active profile exposes; first-run model pull handled automatically; `just chat "..."` sends a prompt and prints the reply.
- Files: `backend/mneme/llm/`, `backend/mneme/config.py`.
- Acceptance: `just chat "who are you?"` returns a response through `LLMClient` on both profiles. Switching profile needs no code change, only env/detection.

GUARDRAIL: Verify §3.2 before closing the phase. No engine SDK imported outside the `llm/` package.

---

### Phase 1: Vault ingestion

**Goal:** Obsidian vault to clean `Document` + `Chunk` records, structure preserved, link graph captured.

**1.1 Markdown parser**
- Tasks: parse YAML frontmatter, `[[wikilinks]]` (including `[[note#heading]]` and `[[note|alias]]`), `#tags`, and heading hierarchy. Emit `Document`.
- Files: `ingestion/mneme_ingest/parser.py`.
- Acceptance: on a fixture vault, frontmatter, tags, and outgoing wikilinks are extracted correctly, including the tricky link forms above.

**1.2 Structural chunker**
- Tasks: split each document along heading boundaries; never split mid-sentence; never merge content across sibling H2/H3 sections; configurable target size with bounded overlap; populate `heading_path` and `token_count`.
- Files: `ingestion/mneme_ingest/chunker.py`.
- Acceptance: no chunk crosses a top-level section boundary; every chunk carries a non-empty `heading_path`; a manual sample of 10 chunks reads as self-contained.

GUARDRAIL: Fixed-size character splitting is rejected. The chunker must be structure-aware. If a note has no headings, fall back to recursive splitting with overlap, not blind slicing.

**1.3 Link-graph extraction**
- Tasks: build a directed graph of note-to-note wikilinks; persist as SQLite (or a serialized adjacency structure). Resolve link targets to document ids; record unresolved links separately.
- Files: `ingestion/mneme_ingest/graph.py`.
- Acceptance: graph node count equals document count; edge count matches resolved wikilinks; unresolved links are logged, not dropped silently.

**1.4 Index CLI**
- Tasks: `python -m mneme_ingest.index --vault <path>` runs parse -> chunk -> graph and prints a stats report (documents, chunks, tokens, links, unresolved links).
- Acceptance: runs over the real vault without crashing on edge-case notes; report numbers are internally consistent.

---

### Phase 2: Minimal end-to-end RAG

**Goal:** first vertical slice. Question in, grounded answer with citations out.

**2.1 Embedding + indexing**
- Tasks: embed chunks with BGE-M3 (dense vectors now; keep sparse output for phase 4); upsert to Qdrant with full `Chunk` payload.
- Files: `ingestion/mneme_ingest/embed.py`, extend `index.py`.
- Acceptance: Qdrant collection holds one point per chunk; payload round-trips the `Chunk` fields.

**2.2 Naive retrieval**
- Tasks: dense top-k cosine search; return chunks with scores; prepend `heading_path` to each chunk's text when assembling context.
- Files: `backend/mneme/retrieval/dense.py`.
- Acceptance: for 5 hand-picked questions, the correct source note appears in top-k.

**2.3 Answer synthesis + API**
- Tasks: prompt assembly (context + question + cite-sources instruction + explicit "say you don't know" instruction); `POST /query` returns `QueryResponse` with `sources`.
- Files: `backend/mneme/rag/`, `backend/mneme/api/`.
- Acceptance: a query about a known note returns a correct answer citing that note.

GUARDRAIL: Out-of-domain test is mandatory. Ask something not in the vault; the system must answer "not found in your notes" and must not fabricate. Add this as a test, do not assume it.

---

### Phase 3: Usable frontend

**Goal:** browser chat with streaming and citations.

**3.1 Streaming endpoint**
- Tasks: `GET /query/stream` over SSE; stream answer tokens, then emit a final `sources` event.
- Files: `backend/mneme/api/`.
- Acceptance: a raw SSE client receives incremental tokens followed by a sources payload.

**3.2 React chat UI**
- Tasks: Vite + TS chat; render streamed tokens; render citations as clickable source cards showing `rel_path`, `heading_path`, and the snippet; session-local history.
- Files: `frontend/`.
- Acceptance: owner runs a real query in the browser and finds it comfortable; citation cards point to the right fragment.

GUARDRAIL: No retrieval or prompt logic in the frontend. UI renders; backend decides.

---

### Phase 4: Advanced retrieval

**Goal:** measurably better retrieval. Both naive and advanced paths must stay alive for phase 5.

**4.1 Hybrid search**
- Tasks: index BGE-M3 sparse vectors alongside dense; run dense and sparse queries; fuse with Reciprocal Rank Fusion.
- Files: `backend/mneme/retrieval/sparse.py`, `hybrid.py`.
- Acceptance: queries with exact tokens (acronyms, code identifiers) that naive mode missed are now retrieved.

**4.2 Reranking**
- Tasks: take top-N (e.g. 20) candidates, score with the cross-encoder, return top-K (e.g. 5).
- Files: `backend/mneme/retrieval/rerank.py`.
- Acceptance: rerank is integrated and can be toggled; `mode` selects naive vs hybrid.

**4.3 Metadata filters**
- Tasks: allow filtering by tag or vault folder via the `filters` field.
- Acceptance: a tag-filtered query only returns chunks carrying that tag.

GUARDRAIL: The naive path must remain callable via `mode="naive"`. Phase 5 compares numbers; both paths must coexist.

---

### Phase 5: Evaluation

**Goal:** quantitative evidence. This is the differentiating phase.

**5.1 Golden dataset**
- Tasks: build 30-50 `(question, expected_answer, expected_source_notes)` items over the real vault; generate candidates with the LLM, then the owner curates by hand. The real set lives locally and is git-ignored (it contains personal knowledge). Commit only a small synthetic sample built over the fixture vault, so the eval harness is reproducible by anyone cloning the repo.
- Files: `eval/golden/` (real set git-ignored), `eval/golden_sample/` (synthetic, committed).
- Acceptance: the real dataset is human-reviewed and not tracked by git; the synthetic sample is committed and runs end to end.

GUARDRAIL: Do not commit the real golden set. Do not ship a purely synthetic, unreviewed golden set as the real evaluation. The owner must curate the real one; the committed sample is only for reproducibility.

**5.2 Metrics harness**
- Tasks: integrate RAGAS (faithfulness, answer relevancy, context precision, context recall); add retrieval metrics (hit rate, MRR); run any `mode` over the golden set.
- Files: `eval/run_eval.py`.
- Acceptance: `just eval` produces reproducible metrics for a given mode.

**5.3 Comparative report**
- Tasks: run naive vs hybrid+rerank; emit a markdown/HTML table; `eval/report.py`.
- Acceptance: the report shows a per-metric comparison; the table is suitable for the README.

GUARDRAIL: Watch for self-judging bias (same model generating and grading). Surface failing cases explicitly, not just averages. Treat suspiciously high scores as a smell to investigate.

---

### Phase 6: Graph-aware retrieval

**Goal:** exploit the wikilink graph; prove the effect against the phase-5 baseline.

**6.1 Graph-expanded retrieval**
- Tasks: load the link graph; after base retrieval, pull chunks from 1-hop neighbor notes of the top notes with a reduced weight.
- Files: `backend/mneme/retrieval/graph.py`.
- Acceptance: `mode="graph"` returns neighbor-expanded context; expansion depth and neighbor weight are configurable.

**6.2 Combined re-scoring**
- Tasks: final score = `alpha*similarity + beta*graph_proximity + gamma*recency`; expose the weights via config.
- Acceptance: weights are tunable; setting beta=gamma=0 reproduces hybrid+rerank exactly (regression guard).

**6.3 Measure**
- Tasks: run the graph mode over the golden set; add a row to the comparative report; write a short honest analysis of where it helps and where it does not.
- Acceptance: a second result row exists with interpretation, including any regressions.

GUARDRAIL: Expansion can inject noise and hurt precision. Report the real numbers. "Helps on conceptual questions, neutral on factual" is an acceptable and valuable result. Do not tune until everything looks like it improves.

---

### Phase 7: Agentic layer

**Goal:** handle questions a single-pass RAG cannot. Every node must justify its latency.

**7.1 LangGraph skeleton**
- Tasks: define the state graph; nodes for query-rewrite, retrieve-decision, retrieve, synthesize.
- Files: `backend/mneme/agent/`.
- Acceptance: `mode="agent"` routes a question through the graph and returns a `QueryResponse`.

**7.2 Query rewriting + routing**
- Tasks: rewrite the user question (resolve pronouns, expand acronyms) before retrieval; a routing node decides whether the vault is needed at all.
- Acceptance: a follow-up question with a pronoun retrieves correctly after rewrite; small-talk skips retrieval.

**7.3 Multi-hop + self-correction**
- Tasks: chain retrieval steps for questions needing multiple notes; if retrieved context is weak, reformulate and retry once before answering (CRAG-lite).
- Acceptance: a multi-hop golden question that failed in single-pass now succeeds; the retry path is exercised by a test.

**7.4 Optional web fallback**
- Tasks: a web-search tool used only when the vault clearly lacks the answer; external sources labeled distinctly in `sources`.
- Acceptance: external answers are visibly marked as external; vault-first behavior is preserved.

GUARDRAIL: Do not agentify for fashion. Measure agent mode on the golden set. If multi-hop does not improve metrics and only adds latency, keep it behind a flag and say so in the PR.

---

### Phase 8: Production and observability

**Goal:** deployed, observable, maintainable system.

**8.1 Langfuse tracing**
- Tasks: self-host Langfuse in compose; instrument each query (question, rewrite, retrieved chunks, final prompt, per-stage latency, tokens); return `trace_id` in responses.
- Files: `backend/mneme/obs/`.
- Acceptance: every query is fully traceable in Langfuse end to end.

**8.2 Incremental indexing**
- Tasks: reindex only notes whose `mtime` changed; optional file-watcher trigger; remove orphaned chunks for deleted notes.
- Acceptance: editing one note reindexes only that note; deleting a note removes its chunks.

**8.3 Production overlay + tunnel + auth**
- Tasks: `docker-compose.prod.yml` overlay adding Langfuse, an auth gateway requiring `AUTH_TOKEN`, and the Cloudflare Tunnel on a subroute; structured logging; healthchecks. The base stack from Phase 0 stays runnable standalone for dev.
- Acceptance: `make run` (dev) works unchanged; `make run-prod` brings up the authenticated, tunneled, traceable stack; `/health` reports each dependency.

GUARDRAIL: Authentication must be real before any public exposure. This is private personal knowledge. Do not expose an unauthenticated instance, even temporarily.

---

## 5. Definition of done (whole project)

- `git clone` + `make run` brings up the full stack on any machine, GPU or CPU, with no code changes.
- The public repo has no secrets and no personal data in its history; secret scanning passes; the Obsidian vault and the real golden set were never committed.
- Every phase merged via its own PR with a decision-focused description.
- README leads with the architecture diagram and the evaluation comparison table.
- `docker compose up` yields a working, authenticated, traceable system.
- Naive, hybrid, graph, and agent modes are all reachable and benchmarked.
- The graph-aware result is reported honestly, including any non-improvements.
- A short demo capture (GIF/video) is embedded in the README.

---

## 6. Command reference (target)

```
make run           # detect GPU -> pick profile -> bring up the whole stack
make run-prod      # production overlay: + langfuse, auth, cloudflare tunnel
SERVING_PROFILE=cpu make run   # force CPU regardless of hardware
just chat "..."    # one-shot LLM call via LLMClient
just index         # ingest the vault
just index --incremental
just eval          # run evaluation for the configured mode
just eval-compare  # naive vs hybrid vs graph vs agent
just lint / fmt / test
```

---

## 7. Implementation order summary

```
P0 repo init + hardening + serving -> public repo, model answers via LLMClient
P1 ingestion                  -> Document/Chunk records + link graph
P2 minimal RAG                -> grounded answers with citations
P3 frontend                   -> streaming browser chat
P4 advanced retrieval         -> hybrid + rerank, naive path kept
P5 evaluation                 -> RAGAS + retrieval metrics + comparison table
P6 graph-aware retrieval      -> neighbor expansion, measured vs baseline
P7 agentic layer              -> rewrite, route, multi-hop, self-correct
P8 production                 -> tracing, incremental index, compose, tunnel, auth
```

Build vertically. Keep every mode alive for measurement. Report numbers honestly. Stop at GUARDRAILs.
