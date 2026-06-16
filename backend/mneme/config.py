"""Centralised configuration.

Two layers merge into one Settings object:
  - .env (gitignored): machine-specific and secret values, loaded by pydantic.
  - config.yaml (versioned): per-profile behaviour defaults.

LLM_BASE_URL / LLM_MODEL / QDRANT_URL follow the same precedence detect.sh uses:
an explicit env value wins; otherwise the URL is resolved to an in-cluster
default (vllm, ollama, qdrant) and the host is rewritten to localhost. The
unset path is taken only by host-side tooling (just chat / serve / index), which
must reach the published ports; container services receive an explicit URL via
compose and so skip the rewrite.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import yaml
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = ROOT_DIR / ".env"
CONFIG_YAML = ROOT_DIR / "config.yaml"

_IN_CLUSTER_HOSTS = {"vllm", "ollama", "qdrant"}
_DEFAULT_QDRANT_URL = "http://qdrant:6333"


def config_yaml_value(profile: str, key: str, path: Path = CONFIG_YAML) -> str:
    """Return profiles.<profile>.<key> from config.yaml, or '' if absent."""
    if not path.exists():
        return ""
    data = yaml.safe_load(path.read_text()) or {}
    return str(data.get("profiles", {}).get(profile, {}).get(key, "") or "")


def to_host_url(url: str) -> str:
    """Rewrite an in-cluster service host (vllm/ollama) to localhost."""
    parsed = urlparse(url)
    if parsed.hostname in _IN_CLUSTER_HOSTS:
        port = f":{parsed.port}" if parsed.port else ""
        parsed = parsed._replace(netloc=f"localhost{port}")
        return urlunparse(parsed)
    return url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    serving_profile: str = ""
    llm_base_url: str = ""
    llm_model: str = ""
    llm_api_key: str = "not-needed"

    qdrant_url: str = ""
    qdrant_collection: str = "mneme"

    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "auto"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "auto"
    rerank_enabled: bool = True

    vault_path: str = ""

    # Comma-separated browser origins allowed to call the API (frontend dev server).
    api_cors_origins: str = "http://localhost:5173"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.api_cors_origins.split(",") if o.strip()]

    @model_validator(mode="after")
    def _resolve(self) -> Settings:
        profile = self.serving_profile or "cpu"
        self.serving_profile = profile
        if not self.llm_base_url:
            self.llm_base_url = to_host_url(config_yaml_value(profile, "llm_base_url"))
        if not self.llm_model:
            self.llm_model = config_yaml_value(profile, "llm_model")
        if not self.qdrant_url:
            self.qdrant_url = to_host_url(_DEFAULT_QDRANT_URL)
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
