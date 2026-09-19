"""Unit tests for Vault OpenAI SDK-backed LLM client."""

from unittest.mock import MagicMock

import pytest

from vault.config import Settings
from vault.llm import LLMClient


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3:8b",
        llm_timeout_seconds=5.0,
        llm_max_retries=2,
    )


def test_payload_preparation(test_settings: Settings) -> None:
    """Test LLMClient instance setup."""
    client = LLMClient(settings=test_settings)
    assert client.model == "llama3:8b"
    assert client.timeout == 5.0
    assert client.max_retries == 2


def test_generate_success(test_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Test successful synchronous response generation with mocked OpenAI client."""
    client = LLMClient(settings=test_settings)

    mock_choice = MagicMock()
    mock_choice.message.content = "Hello! I am Vault local assistant."
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    monkeypatch.setattr(
        client.sync_client.chat.completions,
        "create",
        lambda **kwargs: mock_completion,
    )

    result = client.generate([{"role": "user", "content": "Hi"}])
    assert result == "Hello! I am Vault local assistant."


@pytest.mark.anyio
async def test_generate_async_success(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test successful asynchronous response generation with mocked OpenAI client."""
    client = LLMClient(settings=test_settings)

    mock_choice = MagicMock()
    mock_choice.message.content = "Async response content"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    async def mock_create(**kwargs):
        return mock_completion

    monkeypatch.setattr(client.async_client.chat.completions, "create", mock_create)

    result = await client.generate_async([{"role": "user", "content": "Hi"}])
    assert result == "Async response content"
