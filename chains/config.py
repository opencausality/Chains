"""Chains configuration via Pydantic settings."""

from __future__ import annotations

import logging
from enum import Enum
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GROQ = "groq"
    MISTRAL = "mistral"
    TOGETHER = "together"


_DEFAULT_MODELS: dict[LLMProvider, str] = {
    LLMProvider.OLLAMA: "llama3.1",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20241022",
    LLMProvider.GROQ: "llama-3.1-70b-versatile",
    LLMProvider.MISTRAL: "mistral-large-latest",
    LLMProvider.TOGETHER: "meta-llama/Llama-3-70b-chat-hf",
}


class ChainsSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="CHAINS_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llm_provider: LLMProvider = Field(default=LLMProvider.OLLAMA)
    llm_model: str = Field(default="")
    llm_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    llm_max_retries: int = Field(default=3, ge=1, le=10)

    min_traces: int = Field(default=20, ge=5, description="Minimum traces for causal discovery")
    significance_level: float = Field(default=0.01, ge=0.001, le=0.1)
    log_level: str = Field(default="INFO")

    @property
    def resolved_model(self) -> str:
        base = self.llm_model or _DEFAULT_MODELS.get(self.llm_provider, "llama3.1")
        if self.llm_provider == LLMProvider.OLLAMA and "/" not in base:
            return f"ollama/{base}"
        return base


def configure_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s │ %(name)s │ %(levelname)-7s │ %(message)s"
    logging.basicConfig(format=fmt, datefmt="%H:%M:%S", level=getattr(logging, level.upper()))
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("LiteLLM").setLevel(logging.WARNING)


@lru_cache(maxsize=1)
def get_settings() -> ChainsSettings:
    return ChainsSettings()
