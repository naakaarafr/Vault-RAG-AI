"""Unit and smoke tests for OpenAI SDK-backed LLM client in src/vault/llm/client.py."""

from unittest.mock import MagicMock

import pytest

from vault.config import Settings
from vault.llm.client import LLMClient


@pytest.fixture
def test_settings() -> Settings:
    return Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3:8b",
        llm_timeout_seconds=5.0,
        llm_max_retries=2,
    )


def test_client_initialization(test_settings: Settings) -> None:
    """Verify LLMClient initializes sync and async OpenAI clients with base_url."""
    client = LLMClient(settings=test_settings)
    assert str(client.sync_client.base_url).startswith("http://localhost:11434/v1")
    assert client.model == "llama3:8b"


def test_generate_sync_mocked(test_settings: Settings, monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify synchronous chat generation with mocked OpenAI client response."""
    client = LLMClient(settings=test_settings)

    mock_choice = MagicMock()
    mock_choice.message.content = "Response from local Ollama/vLLM"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    monkeypatch.setattr(
        client.sync_client.chat.completions,
        "create",
        lambda **kwargs: mock_completion,
    )

    result = client.generate(messages=[{"role": "user", "content": "Hello"}])
    assert result == "Response from local Ollama/vLLM"


def test_generate_sync_streaming_mocked(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify synchronous streaming generation yields content tokens."""
    client = LLMClient(settings=test_settings)

    chunk1 = MagicMock()
    chunk1.choices = [MagicMock(delta=MagicMock(content="Hello "))]
    chunk2 = MagicMock()
    chunk2.choices = [MagicMock(delta=MagicMock(content="world!"))]

    monkeypatch.setattr(
        client.sync_client.chat.completions,
        "create",
        lambda **kwargs: [chunk1, chunk2],
    )

    stream_iter = client.generate(
        messages=[{"role": "user", "content": "Hi"}],
        stream=True,
    )

    tokens = list(stream_iter)
    assert tokens == ["Hello ", "world!"]


@pytest.mark.anyio
async def test_generate_async_mocked(
    test_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verify asynchronous chat generation with mocked OpenAI client response."""
    client = LLMClient(settings=test_settings)

    mock_choice = MagicMock()
    mock_choice.message.content = "Async response from local Ollama/vLLM"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    async def mock_create(**kwargs):
        return mock_completion

    monkeypatch.setattr(client.async_client.chat.completions, "create", mock_create)

    result = await client.generate_async(messages=[{"role": "user", "content": "Hello"}])
    assert result == "Async response from local Ollama/vLLM"


def test_generate_sync_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify primary model failure triggers fallback to OpenAI model synchronously."""
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3:8b",
        openai_api_key="sk-test-key",
        openai_model="gpt-4o-mini",
        enable_openai_fallback=True,
    )
    client = LLMClient(settings=settings)

    def mock_primary_fail(**kwargs):
        raise ConnectionError("Local vLLM server unreachable")

    mock_choice = MagicMock()
    mock_choice.message.content = "Response from OpenAI fallback model"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    fallback_called = {}

    def mock_fallback_success(**kwargs):
        fallback_called["model"] = kwargs.get("model")
        return mock_completion

    monkeypatch.setattr(client.sync_client.chat.completions, "create", mock_primary_fail)
    monkeypatch.setattr(client.fallback_sync_client.chat.completions, "create", mock_fallback_success)

    result = client.generate(messages=[{"role": "user", "content": "Hello"}])
    assert result == "Response from OpenAI fallback model"
    assert fallback_called.get("model") == "gpt-4o-mini"


@pytest.mark.anyio
async def test_generate_async_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify primary model failure triggers fallback to OpenAI model asynchronously."""
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3:8b",
        openai_api_key="sk-test-key",
        openai_model="gpt-4o-mini",
        enable_openai_fallback=True,
    )
    client = LLMClient(settings=settings)

    async def mock_primary_fail(**kwargs):
        raise ConnectionError("Local vLLM server unreachable")

    mock_choice = MagicMock()
    mock_choice.message.content = "Async response from OpenAI fallback model"
    mock_completion = MagicMock()
    mock_completion.choices = [mock_choice]

    fallback_called = {}

    async def mock_fallback_success(**kwargs):
        fallback_called["model"] = kwargs.get("model")
        return mock_completion

    monkeypatch.setattr(client.async_client.chat.completions, "create", mock_primary_fail)
    monkeypatch.setattr(client.fallback_async_client.chat.completions, "create", mock_fallback_success)

    result = await client.generate_async(messages=[{"role": "user", "content": "Hello"}])
    assert result == "Async response from OpenAI fallback model"
    assert fallback_called.get("model") == "gpt-4o-mini"


def test_generate_fallback_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify primary model failure raises exception when fallback is disabled."""
    settings = Settings(
        llm_base_url="http://localhost:11434/v1",
        llm_model="llama3:8b",
        openai_api_key="sk-test-key",
        enable_openai_fallback=False,
    )
    client = LLMClient(settings=settings)

    def mock_primary_fail(**kwargs):
        raise ConnectionError("Local vLLM server unreachable")

    monkeypatch.setattr(client.sync_client.chat.completions, "create", mock_primary_fail)

    with pytest.raises(ConnectionError, match="Local vLLM server unreachable"):
        client.generate(messages=[{"role": "user", "content": "Hello"}])

