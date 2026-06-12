set dotenv-load := true

default:
    @just --list

# Apply code formatting
fmt:
    uv run black .
    uv run ruff check --fix .

# Run linting checks (read-only)
lint:
    uv run ruff check .
    uv run black --check .

# Run the test suite
test:
    uv run pytest

# Ingest a vault: parse -> chunk -> graph, print stats (uses $VAULT_PATH if --vault omitted)
index *args:
    uv run python -m mneme_ingest.index {{args}}

# Serve the API locally (builds real BGE-M3 on startup; needs Qdrant + the embed group)
serve:
    uv run uvicorn --factory mneme.api.app:create_app --host 0.0.0.0 --port 8001

# Send a one-shot prompt through LLMClient
chat +prompt:
    uv run python -m mneme.llm.chat "{{prompt}}"
