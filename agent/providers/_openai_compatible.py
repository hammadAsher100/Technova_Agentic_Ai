"""Shared request/response mapping for providers exposing an
OpenAI-compatible `/chat/completions` endpoint: Groq, Mistral, and
OpenRouter. Each of those three client modules is a thin, provider-
specific wrapper around `complete()` below — this is what keeps the
mapping logic in exactly one place (DRY) instead of triplicated across
groq_client.py / mistral_client.py / openrouter_client.py.

Gemini does not use this module: its REST contract (contents/parts,
not messages) is genuinely different — see gemini_client.py.
"""
import time
from typing import Any, Dict, List, Optional

from agent.exceptions import ProviderInvalidResponseError
from agent.providers import _http
from agent.providers.base import LLMRequest, LLMResponse


def _build_payload(request: LLMRequest, model: str) -> Dict[str, Any]:
    messages: List[Dict[str, str]] = []
    if request.system_prompt:
        messages.append({"role": "system", "content": request.system_prompt})
    for msg in request.messages:
        messages.append({"role": msg.role, "content": msg.content})
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
    }
    if request.response_schema:
        payload["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "trust_arena_decision", "strict": True, "schema": request.response_schema}}
    if request.reasoning_effort:
        payload["reasoning_effort"] = request.reasoning_effort
    return payload


def _parse_response(
    raw: Dict[str, Any], provider_name: str, model: str, latency_seconds: float
) -> LLMResponse:
    try:
        choice = raw["choices"][0]
        text = choice["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderInvalidResponseError(
            f"{provider_name}: response missing choices[0].message.content"
        ) from exc
    usage = raw.get("usage") or {}
    return LLMResponse(
        text=text,
        provider=provider_name,
        model=raw.get("model", model),
        latency_seconds=latency_seconds,
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        finish_reason=choice.get("finish_reason"),
        raw=raw,
    )


def complete(
    *,
    provider_name: str,
    base_url: str,
    api_key: str,
    model: str,
    request: LLMRequest,
    timeout: float,
    extra_headers: Optional[Dict[str, str]] = None,
) -> LLMResponse:
    headers = {"Authorization": f"Bearer {api_key}"}
    if extra_headers:
        headers.update(extra_headers)
    payload = _build_payload(request, model)

    start = time.monotonic()
    _, body = _http.post_json(base_url, headers, payload, timeout)
    latency_seconds = time.monotonic() - start

    return _parse_response(body, provider_name, model, latency_seconds)
