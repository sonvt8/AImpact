import importlib

import pytest

import config


ENV_KEYS = (
    "APP_PASSWORD",
    "DATA_DIR",
    "HISTORY_DIR",
    "DOCUMENTS_DIR",
    "MODEL_PATH",
    "SIMILARITY_THRESHOLD",
    "LLM_BACKEND",
    "LLM_BASE_URL",
    "LLM_MODEL",
    "LLM_API_KEY",
    "LLM_TIMEOUT",
    "OLLAMA_BASE_URL",
    "OLLAMA_MODEL",
    "LLM_SKIP_INTERNET_CHECK",
    "EMBEDDING_MODEL_ID",
    "DOMAIN_PROFILE",
)


def reload_config(monkeypatch, **env):
    monkeypatch.setattr("dotenv.load_dotenv", lambda: None)
    for key in ENV_KEYS:
        monkeypatch.delenv(key, raising=False)
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    return importlib.reload(config)


def test_env_overrides_and_openai_alias(monkeypatch):
    cfg = reload_config(
        monkeypatch,
        APP_PASSWORD="secret",
        LLM_BACKEND="openai_compatible",
        LLM_BASE_URL="http://9router.local/v1",
        LLM_MODEL="gpt-4o-mini",
        LLM_TIMEOUT="45",
        SIMILARITY_THRESHOLD="0.75",
    )

    cfg.validate()
    assert cfg.APP_PASSWORD == "secret"
    assert cfg.SIMILARITY_THRESHOLD == 0.75
    assert cfg.build_llm_client_kwargs("openai") == cfg.build_llm_client_kwargs(
        "openai_compatible"
    ) == {
        "model": "gpt-4o-mini",
        "base_url": "http://9router.local/v1",
        "api_key": "not-needed",
        "timeout": 45,
        "temperature": 0.1,
        "streaming": True,
    }


def test_validate_requires_app_password(monkeypatch):
    cfg = reload_config(monkeypatch, APP_PASSWORD="", LLM_BACKEND="ollama")

    with pytest.raises(ValueError, match="APP_PASSWORD"):
        cfg.validate()


def test_ollama_defaults(monkeypatch):
    cfg = reload_config(monkeypatch, APP_PASSWORD="secret", LLM_BACKEND="ollama")

    cfg.validate()
    assert cfg.LLM_SKIP_INTERNET_CHECK is True
    assert cfg.build_llm_client_kwargs("ollama") == {
        "model": "llama3.2",
        "base_url": "http://localhost:11434",
        "temperature": 0.1,
        "streaming": True,
    }
