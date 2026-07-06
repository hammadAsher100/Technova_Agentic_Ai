"""Mistral provider client (transport-only).

See groq_client.py's module docstring for the full rationale: this
calls Mistral's OpenAI-compatible REST endpoint directly instead of the
official `mistralai` SDK, which currently requires Python >=3.10
(REG-02 requires 3.9 compatibility).
"""
from agent.providers import _openai_compatible
from agent.providers.base import BaseProviderClient, LLMRequest, LLMResponse

_MISTRAL_BASE_URL = "https://api.mistral.ai/v1/chat/completions"


class MistralClient(BaseProviderClient):
    provider_id = "mistral"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("MistralClient requires a non-empty api_key")
        if not model:
            raise ValueError("MistralClient requires a non-empty model name")
        self._api_key = api_key
        self._model = model

    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        return _openai_compatible.complete(
            provider_name=self.provider_id,
            base_url=_MISTRAL_BASE_URL,
            api_key=self._api_key,
            model=self._model,
            request=request,
            timeout=timeout,
        )
