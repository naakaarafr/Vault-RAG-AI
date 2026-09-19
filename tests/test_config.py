"""Unit tests for Vault configuration management."""

import pytest

from vault.config import Settings, get_settings


def test_default_settings() -> None:
    """Verify default setting values and types."""
    settings = Settings()
    assert settings.llm_base_url.startswith("http")
    assert isinstance(settings.llm_timeout_seconds, float)
    assert settings.llm_max_retries > 0
    assert isinstance(settings.enable_input_guardrails, bool)


def test_custom_settings_override(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify environment variable overrides work cleanly."""
    monkeypatch.setenv("LLM_BASE_URL", "http://localhost:8000/v1")
    monkeypatch.setenv("LLM_MODEL", "vllm-custom")
    monkeypatch.setenv("ENV", "testing")

    settings = Settings()
    assert settings.llm_base_url == "http://localhost:8000/v1"
    assert settings.llm_model == "vllm-custom"
    assert settings.env == "testing"


def test_get_settings_caching() -> None:
    """Verify get_settings returns a cached instance."""
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
