"""Gemini provider client (transport-only), implemented via the Gemini
REST API's `generateContent` endpoint using the Python standard
library.

WHY NOT THE OFFICIAL SDK — see groq_client.py's module docstring for
the full rationale (`google-genai` currently requires Python >=3.10;
REG-02 requires 3.9 compatibility).

One class backs two ModelRouter profiles (Section 5: Gemini Flash and
Gemini Pro are separate, separately-prioritized entries) — construct
two instances with different `model` and `provider_id`.
"""
import time
from typing import Any, Dict

from agent.exceptions import ProviderInvalidResponseError
from agent.providers import _http
from agent.providers.base import BaseProviderClient, LLMRequest, LLMResponse

_GEMINI_BASE_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)


class GeminiClient(BaseProviderClient):
    def __init__(self, api_key: str, model: str, provider_id: str = "gemini") -> None:
        if not api_key:
            raise ValueError("GeminiClient requires a non-empty api_key")
        if not model:
            raise ValueError("GeminiClient requires a non-empty model name")
        self._api_key = api_key
        self._model = model
        self.provider_id = provider_id

    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        url = _GEMINI_BASE_URL.format(model=self._model)
        headers = {"x-goog-api-key": self._api_key}
        payload = self._build_payload(request)

        start = time.monotonic()
        _, body = _http.post_json(url, headers, payload, timeout)
        latency_seconds = time.monotonic() - start

        return self._parse_response(body, latency_seconds)

    def _build_payload(self, request: LLMRequest) -> Dict[str, Any]:
        contents = []
        for msg in request.messages:
            role = "model" if msg.role == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg.content}]})
        payload: Dict[str, Any] = {
            "contents": contents,
            "generationConfig": {
                "maxOutputTokens": request.max_tokens,
                "temperature": request.temperature,
            },
        }
        if request.response_schema:
            payload["generationConfig"]["responseMimeType"] = "application/json"
            payload["generationConfig"]["responseJsonSchema"] = request.response_schema
        if request.system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": request.system_prompt}]}
        return payload

    def _parse_response(self, raw: Dict[str, Any], latency_seconds: float) -> LLMResponse:
        try:
            candidate = raw["candidates"][0]
            parts = candidate["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as exc:
            raise ProviderInvalidResponseError(
                f"{self.provider_id}: response missing candidates[0].content.parts"
            ) from exc
        usage = raw.get("usageMetadata") or {}
        return LLMResponse(
            text=text,
            provider=self.provider_id,
            model=self._model,
            latency_seconds=latency_seconds,
            input_tokens=usage.get("promptTokenCount"),
            output_tokens=usage.get("candidatesTokenCount"),
            finish_reason=candidate.get("finishReason"),
            raw=raw,
        )
