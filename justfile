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

# Bring up the full stack (GPU/CPU auto-detected)
run:
    bash scripts/detect.sh

# Send a one-shot prompt through LLMClient
chat +prompt:
    uv run python -m mneme.llm.chat "{{prompt}}"
