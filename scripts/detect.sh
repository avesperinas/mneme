#!/usr/bin/env bash
# Hardware-agnostic launcher for the Mneme base stack.
#
# Resolves a serving profile (gpu | cpu) and the matching LLM endpoint, then
# brings up the compose stack for that profile. Profile precedence:
#   1. SERVING_PROFILE in the environment (e.g. `SERVING_PROFILE=cpu make run`)
#   2. SERVING_PROFILE in .env
#   3. Auto-detect: gpu only if Docker can actually use an NVIDIA GPU, else cpu
#
# A missing GPU never hard-fails; it falls back to the cpu (Ollama) profile.
# Set MNEME_DRY_RUN=1 to print the resolved settings and compose command
# without pulling images or starting containers.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CONFIG_FILE="$ROOT_DIR/config.yaml"
ENV_FILE="$ROOT_DIR/.env"

log() { printf '[detect] %s\n' "$*" >&2; }

# Read a top-level KEY=value from .env, ignoring comments. Empty if absent.
read_env() {
    [ -f "$ENV_FILE" ] || return 0
    grep -E "^$1=" "$ENV_FILE" 2>/dev/null \
        | tail -n1 | cut -d= -f2- | tr -d '"' | xargs || true
}

# Extract a nested value for a profile from config.yaml (no yaml dependency).
# Usage: get_config_value <profile> <key>
get_config_value() {
    awk -v profile="$1" -v key="$2" '
        /^  [A-Za-z0-9_]+:[[:space:]]*$/ {
            in_profile = ($0 ~ ("^  " profile ":[[:space:]]*$")) ? 1 : 0
            next
        }
        /^[A-Za-z0-9_]+:/ { in_profile = 0 }
        in_profile && $0 ~ ("^    " key ":") {
            line = $0
            sub(/^[^:]*:[[:space:]]*/, "", line)
            gsub(/"/, "", line)
            sub(/[[:space:]]+$/, "", line)
            print line
            exit
        }
    ' "$CONFIG_FILE"
}

# True only when Docker itself can claim an NVIDIA GPU, not merely when the
# host has nvidia-smi. Checks the docker runtime first; falls back to a real
# `--gpus all` probe, gated on nvidia-smi so CPU-only hosts skip the image pull.
has_docker_gpu() {
    command -v docker >/dev/null 2>&1 || return 1
    if docker info 2>/dev/null | grep -qiE 'Runtimes:.*nvidia'; then
        return 0
    fi
    if command -v nvidia-smi >/dev/null 2>&1; then
        if docker run --rm --gpus all --entrypoint /bin/true \
            nvidia/cuda:12.4.0-base-ubuntu22.04 >/dev/null 2>&1; then
            return 0
        fi
    fi
    return 1
}

# --- Resolve profile -------------------------------------------------------
PROFILE="${SERVING_PROFILE:-}"
[ -z "$PROFILE" ] && PROFILE="$(read_env SERVING_PROFILE)"

if [ -z "$PROFILE" ]; then
    if has_docker_gpu; then
        PROFILE="gpu"
        log "auto-detected usable NVIDIA GPU -> profile=gpu"
    else
        PROFILE="cpu"
        log "no usable Docker GPU -> profile=cpu"
    fi
else
    log "profile override -> profile=$PROFILE"
fi

if [ "$PROFILE" != "gpu" ] && [ "$PROFILE" != "cpu" ]; then
    log "ERROR: SERVING_PROFILE must be 'gpu' or 'cpu', got '$PROFILE'"
    exit 1
fi

# --- Resolve LLM endpoint --------------------------------------------------
# Precedence: environment > .env > config.yaml profile default.
LLM_BASE_URL="${LLM_BASE_URL:-$(read_env LLM_BASE_URL)}"
[ -z "$LLM_BASE_URL" ] && LLM_BASE_URL="$(get_config_value "$PROFILE" llm_base_url)"
LLM_MODEL="${LLM_MODEL:-$(read_env LLM_MODEL)}"
[ -z "$LLM_MODEL" ] && LLM_MODEL="$(get_config_value "$PROFILE" llm_model)"

export SERVING_PROFILE="$PROFILE"
export LLM_BASE_URL
export LLM_MODEL

log "SERVING_PROFILE=$PROFILE"
log "LLM_BASE_URL=$LLM_BASE_URL"
log "LLM_MODEL=$LLM_MODEL"

if [ "$PROFILE" = "gpu" ]; then
    log "First run downloads the model weights into a named volume; expect a wait of several minutes."
else
    log "First run pulls the Ollama model on first query; expect a short wait."
fi

# --- Launch ----------------------------------------------------------------
if [ -n "${MNEME_DRY_RUN:-}" ]; then
    log "dry run: docker compose --profile $PROFILE up -d"
    exit 0
fi

cd "$ROOT_DIR"
docker compose --profile "$PROFILE" up -d
