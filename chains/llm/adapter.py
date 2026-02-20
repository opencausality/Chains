"""LiteLLM adapter for Chains meta-analysis."""

from __future__ import annotations

import logging
import time
from typing import Any

import litellm

from chains.config import ChainsSettings, get_settings

logger = logging.getLogger(__name__)


class ChainsProviderError(Exception):
    pass


class LLMAdapter:
    def __init__(self, settings: ChainsSettings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model = self._settings.resolved_model

    def complete(self, prompt: str, *, system: str | None = None,
                 temperature: float | None = None, max_tokens: int = 4096,
                 format_json: bool = False, **kwargs: Any) -> str:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        temp = temperature if temperature is not None else self._settings.llm_temperature
        call_kwargs = {**kwargs}
        if format_json:
            call_kwargs["response_format"] = {"type": "json_object"}
        for attempt in range(1, self._settings.llm_max_retries + 1):
            try:
                response = litellm.completion(model=self._model, messages=messages,
                                              temperature=temp, max_tokens=max_tokens, **call_kwargs)
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt < self._settings.llm_max_retries:
                    time.sleep(2 ** (attempt - 1))
        raise ChainsProviderError(f"All LLM attempts failed.")

    @property
    def provider_info(self) -> dict[str, str]:
        return {"provider": self._settings.llm_provider.value, "model": self._model}
