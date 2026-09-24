"""Thin OpenAI SDK wrapper for local OpenAI-compatible chat API (vLLM, Ollama) with OpenAI model fallback."""

import logging
from collections.abc import AsyncIterator, Iterator
from typing import Any

from openai import AsyncOpenAI, OpenAI

from vault.config import Settings, get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Wrapper over OpenAI-compatible chat endpoint using standard openai SDK with fallback to OpenAI model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.base_url = self.settings.llm_base_url
        self.model = self.settings.llm_model
        self.timeout = self.settings.llm_timeout_seconds
        self.max_retries = self.settings.llm_max_retries

        # OpenAI fallback configuration
        self.openai_api_key = self.settings.openai_api_key
        self.openai_model = self.settings.openai_model
        self.openai_base_url = self.settings.openai_base_url
        self.enable_openai_fallback = self.settings.enable_openai_fallback

        # Initialize primary synchronous and asynchronous OpenAI clients pointing to local server
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

        self._fallback_sync_client: OpenAI | None = None
        self._fallback_async_client: AsyncOpenAI | None = None

    @property
    def fallback_sync_client(self) -> OpenAI | None:
        """Lazy-loaded synchronous client for OpenAI fallback model."""
        if self._fallback_sync_client is None and self.openai_api_key:
            self._fallback_sync_client = OpenAI(
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._fallback_sync_client

    @property
    def fallback_async_client(self) -> AsyncOpenAI | None:
        """Lazy-loaded asynchronous client for OpenAI fallback model."""
        if self._fallback_async_client is None and self.openai_api_key:
            self._fallback_async_client = AsyncOpenAI(
                base_url=self.openai_base_url,
                api_key=self.openai_api_key,
                timeout=self.timeout,
                max_retries=self.max_retries,
            )
        return self._fallback_async_client

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | Iterator[str]:
        """Send chat completion request to primary LLM server with fallback to OpenAI model."""
        kwargs_clean = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs_clean["max_tokens"] = max_tokens
        kwargs_clean.update(kwargs)

        try:
            if stream:
                response_stream = self.sync_client.chat.completions.create(**kwargs_clean)
                return self._stream_generator(response_stream)

            response = self.sync_client.chat.completions.create(**kwargs_clean)
            return response.choices[0].message.content or ""
        except Exception as exc:
            if self.enable_openai_fallback and self.fallback_sync_client:
                logger.warning(
                    f"Primary LLM model '{self.model}' request failed with error: {exc}. "
                    f"Falling back to OpenAI model '{self.openai_model}'."
                )
                fallback_kwargs = dict(kwargs_clean)
                fallback_kwargs["model"] = self.openai_model
                if stream:
                    response_stream = self.fallback_sync_client.chat.completions.create(**fallback_kwargs)
                    return self._stream_generator(response_stream)

                response = self.fallback_sync_client.chat.completions.create(**fallback_kwargs)
                return response.choices[0].message.content or ""
            raise

    async def generate_async(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int | None = 1024,
        stream: bool = False,
        **kwargs: Any,
    ) -> str | AsyncIterator[str]:
        """Asynchronously send chat completion request to primary LLM server with fallback to OpenAI model."""
        kwargs_clean = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }
        if max_tokens is not None:
            kwargs_clean["max_tokens"] = max_tokens
        kwargs_clean.update(kwargs)

        try:
            if stream:
                response_stream = await self.async_client.chat.completions.create(**kwargs_clean)
                return self._async_stream_generator(response_stream)

            response = await self.async_client.chat.completions.create(**kwargs_clean)
            return response.choices[0].message.content or ""
        except Exception as exc:
            if self.enable_openai_fallback and self.fallback_async_client:
                logger.warning(
                    f"Primary LLM model '{self.model}' async request failed with error: {exc}. "
                    f"Falling back to OpenAI model '{self.openai_model}'."
                )
                fallback_kwargs = dict(kwargs_clean)
                fallback_kwargs["model"] = self.openai_model
                if stream:
                    response_stream = await self.fallback_async_client.chat.completions.create(**fallback_kwargs)
                    return self._async_stream_generator(response_stream)

                response = await self.fallback_async_client.chat.completions.create(**fallback_kwargs)
                return response.choices[0].message.content or ""
            raise

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

