# Mneme

An agentic retrieval-augmented generation (RAG) system over a personal Obsidian knowledge base, self-hosted with GPU/CPU-adaptive LLM serving.

## Architecture

*(Architecture diagram and evaluation comparison table will be added in later phases.)*

Core components:

- **Ingestion** -- Obsidian markdown parser, structural chunker, wikilink graph extraction
- **Retrieval** -- BGE-M3 dense + sparse vectors in Qdrant, reranked with BGE-reranker-v2-m3
- **Graph layer** -- wikilink-graph neighbor expansion for multi-hop questions
- **LLM serving** -- vLLM (GPU) or Ollama (CPU), auto-detected; both via OpenAI-compatible API
- **Agent** -- LangGraph state graph: query rewrite, retrieval, self-correction, multi-hop
- **Frontend** -- React + Vite streaming chat with citation cards
- **Observability** -- self-hosted Langfuse tracing

## Quick start

### Prerequisites

- Docker + Docker Compose v2
- `make`
- An Obsidian vault accessible on the host

### Setup

```bash
git clone <repo-url>
cd mneme
cp .env.example .env
# Edit .env: set VAULT_PATH to your vault's absolute path.
# SERVING_PROFILE is auto-detected (gpu or cpu); override if needed.
make run
```

On first run, model weights are downloaded into named Docker volumes. Depending on your connection, this can take 10-30 minutes. Subsequent starts are fast.

### Hardware profiles

| Profile | Serving engine | Default model | Trigger |
|---|---|---|---|
| `gpu` | vLLM | Qwen2.5-7B-Instruct-AWQ | NVIDIA GPU visible to Docker |
| `cpu` | Ollama | qwen2.5:3b | no GPU, or `SERVING_PROFILE=cpu` |

The system always falls back to CPU if no GPU is available.

## Development

```bash
# Install dev tools
pip install pre-commit
pre-commit install      # wire secret scanning into git hooks

# Lint and test
just lint
just test

# Send a one-shot prompt through LLMClient
just chat "What is the capital of France?"
```

## Commands

| Command | Description |
|---|---|
| `make run` | Detect GPU, select profile, bring up the full stack |
| `make run-prod` | Production overlay (Langfuse, auth, Cloudflare Tunnel) |
| `SERVING_PROFILE=cpu make run` | Force CPU regardless of hardware |
| `just index` | Ingest the vault |
| `just index --incremental` | Reindex changed notes only |
| `just eval` | Run evaluation for the configured mode |
| `just eval-compare` | Compare naive vs hybrid vs graph vs agent |
| `just lint` | Run ruff + black |
| `just test` | Run the test suite |

## License

[MIT](LICENSE)
