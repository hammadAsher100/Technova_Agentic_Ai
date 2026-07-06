"""Groq provider client (transport-only) — Section 5's fastest, first
choice for time-sensitive decisions.

Implemented as a direct HTTPS call to Groq's OpenAI-compatible
`/openai/v1/chat/completions` endpoint via the Python standard library,
rather than the official `groq` SDK.

WHY NOT THE OFFICIAL SDK — READ BEFORE "FIXING" THIS:
As of Phase 0 planning, `pip install groq` resolves to a release whose
PyPI metadata declares `Requires-Python: >=3.10`. REG-02 requires this
codebase to run on a judging environment that may only provide Python
3.9. Depending on that SDK risks a hard `pip install` failure on the
judges' machine. Full rationale: docs/ARCHITECTURE.md.

If your team CONFIRMS the judging environment is Python 3.10+, this is
a contained, single-file swap: reimplement `GroqClient.complete()`
using `from groq import Groq` instead, keeping the same method
signature so nothing else in the codebase needs to change (see
agent/providers/base.py for the interface this class must satisfy).
"""
from agent.providers import _openai_compatible
from agent.providers.base import BaseProviderClient, LLMRequest, LLMResponse

_GROQ_BASE_URL = "https://api.groq.com/openai/v1/chat/completions"


class GroqClient(BaseProviderClient):
    provider_id = "groq"

    def __init__(self, api_key: str, model: str) -> None:
        if not api_key:
            raise ValueError("GroqClient requires a non-empty api_key")
        if not model:
            raise ValueError("GroqClient requires a non-empty model name")
        self._api_key = api_key
        self._model = model

    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        return _openai_compatible.complete(
            provider_name=self.provider_id,
            base_url=_GROQ_BASE_URL,
            api_key=self._api_key,
            model=self._model,
            request=request,
            timeout=timeout,
        )
