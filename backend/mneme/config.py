"""Centralised configuration.

Two layers merge into one Settings object:
  - .env (gitignored): machine-specific and secret values, loaded by pydantic.
  - config.yaml (versioned): per-profile behaviour defaults.

LLM_BASE_URL / LLM_MODEL follow the same precedence detect.sh uses:
explicit env value wins; otherwise they are resolved from config.yaml for the
active profile. The config.yaml endpoints use in-cluster service names
(vllm, ollama); when no explicit URL is given we rewrite the host to localhost,
because the only consumer of the unset path is host-side tooling such as
`just chat`. Container services receive an explicit LLM_BASE_URL and so skip
the rewrite.
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

_IN_CLUSTER_HOSTS = {"vllm", "ollama"}


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

    qdrant_url: str = "http://qdrant:6333"
    qdrant_collection: str = "mneme"

    embed_model: str = "BAAI/bge-m3"
    embed_device: str = "auto"
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_device: str = "auto"

    vault_path: str = ""

    @model_validator(mode="after")
    def _resolve_llm(self) -> Settings:
        profile = self.serving_profile or "cpu"
        self.serving_profile = profile
        if not self.llm_base_url:
            self.llm_base_url = to_host_url(config_yaml_value(profile, "llm_base_url"))
        if not self.llm_model:
            self.llm_model = config_yaml_value(profile, "llm_model")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
