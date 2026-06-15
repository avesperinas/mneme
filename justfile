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
    uv run uvicorn --factory mneme.api.app:create_app \
        --host ${API_HOST:-0.0.0.0} --port ${API_PORT:-8001}

# Ask the running API a question (POST /query at $API_BASE_URL)
ask +question:
    #!/usr/bin/env bash
    set -euo pipefail
    url="${API_BASE_URL:-http://localhost:8001}/query"
    payload="$(python3 -c 'import json,sys; print(json.dumps({"question": sys.argv[1]}))' "{{question}}")"
    body="$(mktemp)"; trap 'rm -f "$body"' EXIT
    if ! code="$(curl -sS --connect-timeout 3 --max-time 180 -o "$body" -w '%{http_code}' \
        "$url" -H 'content-type: application/json' --data "$payload")"; then
        echo "Request to $url failed or timed out. Is the API up (just serve) and reachable?" >&2
        exit 1
    fi
    if [ "$code" = 200 ]; then
        python3 -m json.tool "$body"
    else
        echo "HTTP $code from $url:" >&2
        cat "$body" >&2; echo >&2
        exit 1
    fi

# Send a one-shot prompt through LLMClient
chat +prompt:
    uv run python -m mneme.llm.chat "{{prompt}}"
