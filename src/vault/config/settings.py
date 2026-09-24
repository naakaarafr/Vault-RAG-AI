"""Configuration management for Vault using pydantic-settings."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Central settings model loaded from environment variables and .env file."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Local LLM Server Configuration
    llm_base_url: str = Field(
        default="http://localhost:11434/v1",
        description="Base URL for OpenAI-compatible local server (vLLM or Ollama)",
    )
    llm_model: str = Field(
        default="llama3:8b",
        description="Local model name served by vLLM or Ollama",
    )
    llm_timeout_seconds: float = Field(
        default=60.0,
        description="Timeout in seconds for LLM HTTP requests",
    )
    llm_max_retries: int = Field(
        default=3,
        description="Maximum number of HTTP retries on LLM server failure",
    )

    # OpenAI Fallback Model Configuration
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key for fallback OpenAI model calls",
    )
    openai_model: str = Field(
        default="gpt-4o-mini",
        description="OpenAI model name to use when primary local LLM fails",
    )
    openai_base_url: str = Field(
        default="https://api.openai.com/v1",
        description="Base URL for OpenAI API fallback endpoint",
    )
    enable_openai_fallback: bool = Field(
        default=False,
        description="Whether to fall back to OpenAI model if primary local LLM call fails",
    )

    # Models Configuration
    embed_model: str = Field(
        default="BAAI/bge-small-en-v1.5",
        description="Embedding model name or path",
    )
    rerank_model: str = Field(
        default="BAAI/bge-reranker-base",
        description="Reranker model name or path",
    )

    # Infrastructure Services & Storage
    vector_backend: Literal["qdrant", "embedded"] = Field(
        default="embedded",
        description="Vector storage backend engine choice ('qdrant' or 'embedded')",
    )
    qdrant_url: str = Field(
        default="http://localhost:6333",
        description="Local Qdrant vector database URL",
    )
    postgres_dsn: str = Field(
        default="postgresql://vault:vault@localhost:5432/vault",
        description="PostgreSQL connection DSN",
    )
    data_dir: str = Field(
        default="./data",
        description="Local data directory path",
    )

    # Guardrails Configuration
    enable_input_guardrails: bool = Field(
        default=True,
        description="Whether to enable input prompt injection and PII sanitization",
    )
    enable_output_guardrails: bool = Field(
        default=True,
        description="Whether to enable output groundedness and format validation",
    )
    pii_redaction_enabled: bool = Field(
        default=True,
        description="Redact PII in user queries prior to LLM submission",
    )

    # Environment & Logging
    env: Literal["development", "testing", "production"] = Field(
        default="development",
        description="Runtime environment name",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level threshold",
    )


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of Vault Settings."""
    return Settings()
