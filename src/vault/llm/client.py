"""Thin OpenAI SDK wrapper for local OpenAI-compatible chat API (vLLM, Ollama)."""

from collections.abc import AsyncIterator, Iterator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from vault.config import Settings, get_settings


class LLMClient:
    """Wrapper over OpenAI-compatible chat endpoint using standard openai SDK."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm_base_url
        self.model = self.settings.llm_model
        self.timeout = self.settings.llm_timeout_seconds
        self.max_retries = self.settings.llm_max_retries

        # Initialize synchronous and asynchronous OpenAI clients pointing to local server
        self.sync_client = OpenAI(
            base_url=self.base_url,
            api_key="local-vault",  # Placeholder required by SDK for local endpoints
            timeout=self.timeout,
            max_retries=self.max_retries,
        )
        self.async_client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="local-vault",
            timeout=self.timeout,
            max_retries=self.max_retries,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """Send chat completion request to vLLM / Ollama server."""
        kwargs_clean = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs_clean["max_tokens"] = max_tokens
        kwargs_clean.update(kwargs)

        if stream:
            response_stream = self.sync_client.chat.completions.create(**kwargs_clean)
            return self._stream_generator(response_stream)

        response = self.sync_client.chat.completions.create(**kwargs_clean)
        return response.choices[0].message.content or ""

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """Asynchronously send chat completion request to vLLM / Ollama server."""
        kwargs_clean = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs_clean["max_tokens"] = max_tokens
        kwargs_clean.update(kwargs)

        if stream:
            response_stream = await self.async_client.chat.completions.create(**kwargs_clean)
            return self._async_stream_generator(response_stream)

        response = await self.async_client.chat.completions.create(**kwargs_clean)
        return response.choices[0].message.content or ""

    @staticmethod
    def _stream_generator(stream_obj: Any) -> Iterator[str]:
        """Yield content tokens from sync response stream."""
        for chunk in stream_obj:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    @staticmethod
    async def _async_stream_generator(async_stream_obj: Any) -> AsyncIterator[str]:
        """Yield content tokens from async response stream."""
        async for chunk in async_stream_obj:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
