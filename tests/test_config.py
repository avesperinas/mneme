from mneme.config import Settings, config_yaml_value, to_host_url


def test_config_yaml_resolves_profile_defaults():
    assert config_yaml_value("cpu", "llm_base_url") == "http://ollama:11434/v1"
    assert config_yaml_value("cpu", "llm_model") == "qwen2.5:3b"
    assert config_yaml_value("gpu", "llm_base_url") == "http://vllm:8000/v1"
    assert config_yaml_value("gpu", "llm_model") == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_config_yaml_missing_key_is_empty():
    assert config_yaml_value("cpu", "nope") == ""
    assert config_yaml_value("nonexistent", "llm_model") == ""


def test_to_host_url_rewrites_in_cluster_hosts():
    assert to_host_url("http://ollama:11434/v1") == "http://localhost:11434/v1"
    assert to_host_url("http://vllm:8000/v1") == "http://localhost:8000/v1"
    assert to_host_url("http://qdrant:6333") == "http://localhost:6333"


def test_to_host_url_leaves_external_hosts_untouched():
    assert to_host_url("http://example.com:9000/v1") == "http://example.com:9000/v1"
    assert to_host_url("http://localhost:8000/v1") == "http://localhost:8000/v1"


def test_settings_empty_profile_defaults_to_cpu_localhost(monkeypatch):
    for key in ("SERVING_PROFILE", "LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None)
    assert settings.serving_profile == "cpu"
    assert settings.llm_base_url == "http://localhost:11434/v1"
    assert settings.llm_model == "qwen2.5:3b"


def test_settings_gpu_profile_resolves_gpu_defaults(monkeypatch):
    for key in ("LLM_BASE_URL", "LLM_MODEL"):
        monkeypatch.delenv(key, raising=False)
    settings = Settings(_env_file=None, serving_profile="gpu")
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_model == "Qwen/Qwen2.5-7B-Instruct-AWQ"


def test_explicit_base_url_overrides_and_skips_rewrite():
    settings = Settings(
        _env_file=None,
        serving_profile="cpu",
        llm_base_url="http://ollama:11434/v1",
        llm_model="custom",
    )
    # An explicit value is honoured verbatim, no localhost rewrite.
    assert settings.llm_base_url == "http://ollama:11434/v1"
    assert settings.llm_model == "custom"


def test_empty_qdrant_url_resolves_to_localhost(monkeypatch):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    settings = Settings(_env_file=None)
    # host tooling reaches the published port, not the in-cluster name
    assert settings.qdrant_url == "http://localhost:6333"


def test_explicit_qdrant_url_is_honoured():
    settings = Settings(_env_file=None, qdrant_url="http://qdrant:6333")
    assert settings.qdrant_url == "http://qdrant:6333"
