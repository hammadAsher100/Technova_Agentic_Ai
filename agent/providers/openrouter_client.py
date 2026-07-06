"""OpenRouter provider client (transport-only) — Section 5's last-resort
fallback before local-only mode.

OpenRouter's API is itself OpenAI-compatible, so this needs no
OpenRouter-specific SDK either way; see groq_client.py's module
docstring for why the transport is raw HTTP rather than the `openai`
package (also currently Python >=3.10 in its latest releases, and
unnecessary here regardless — see docs/ARCHITECTURE.md).
"""
from typing import Dict, Optional

from agent.providers import _openai_compatible
from agent.providers.base import BaseProviderClient, LLMRequest, LLMResponse

_OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterClient(BaseProviderClient):
    provider_id = "openrouter"

    def __init__(
        self,
        api_key: str,
        model: str,
        site_url: Optional[str] = None,
        site_name: Optional[str] = None,
    ) -> None:
        if not api_key:
            raise ValueError("OpenRouterClient requires a non-empty api_key")
        if not model:
            raise ValueError(
                "OpenRouterClient requires an explicit model — OpenRouter's "
                "free-tier catalog changes too often for a safe hardcoded "
                "default. Set OPENROUTER_MODEL in .env."
            )
        self._api_key = api_key
        self._model = model
        # OpenRouter-recommended (not required) attribution headers.
        extra_headers: Dict[str, str] = {}
        if site_url:
            extra_headers["HTTP-Referer"] = site_url
        if site_name:
            extra_headers["X-Title"] = site_name
        self._extra_headers = extra_headers or None

    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        return _openai_compatible.complete(
            provider_name=self.provider_id,
            base_url=_OPENROUTER_BASE_URL,
            api_key=self._api_key,
            model=self._model,
            request=request,
            timeout=timeout,
            extra_headers=self._extra_headers,
        )
