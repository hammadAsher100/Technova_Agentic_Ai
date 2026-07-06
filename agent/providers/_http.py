"""Minimal, dependency-free HTTP transport built on the Python 3.9
standard library (urllib.request).

WHY NOT `requests`: as of Phase 0 planning (mid-2026), `requests`
itself now requires Python >=3.10, having dropped 3.9 support after its
October 2025 end-of-life — along with every other relevant package
this project would otherwise depend on (groq, google-genai, mistralai,
tenacity). REG-02 requires this codebase to run on a judging
environment that may only have Python 3.9. `urllib.request` ships with
every CPython 3.9+ interpreter, so it is the only HTTP transport this
project can guarantee will import successfully without pinning an
unmaintained, pre-EOL-drop version of a third-party package. Full
rationale: docs/ARCHITECTURE.md, "Dependency strategy".
"""
import json
import socket
import urllib.error
import urllib.request
from typing import Any, Dict, Optional, Tuple

from agent.exceptions import (
    ProviderAuthError,
    ProviderInvalidResponseError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)

_USER_AGENT = "agent-project/1.0 (+phase0-scaffold; stdlib-transport)"


def _safe_json(raw: bytes) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(raw.decode("utf-8"))
    except (ValueError, UnicodeDecodeError):
        return None


def post_json(
    url: str,
    headers: Dict[str, str],
    payload: Dict[str, Any],
    timeout: float,
) -> Tuple[int, Dict[str, Any]]:
    """POST a JSON payload and return (status_code, parsed_json_body).

    Raises a ProviderError subclass on any transport-, auth-, or
    rate-limit-level failure so callers (ModelRouter, provider clients)
    can react without needing to know urllib's exception hierarchy.
    """
    body = json.dumps(payload).encode("utf-8")
    request_headers = {"Content-Type": "application/json", "User-Agent": _USER_AGENT}
    request_headers.update(headers)
    request = urllib.request.Request(url, data=body, headers=request_headers, method="POST")

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read()
            status = response.getcode()
    except urllib.error.HTTPError as exc:
        raw = exc.read()
        parsed = _safe_json(raw)
        status = exc.code
        if status in (401, 403):
            raise ProviderAuthError(f"HTTP {status} from {url}: {parsed}") from exc
        if status == 429:
            raise ProviderRateLimitError(f"HTTP 429 from {url}: {parsed}") from exc
        raise ProviderInvalidResponseError(f"HTTP {status} from {url}: {parsed}") from exc
    except socket.timeout as exc:
        raise ProviderTimeoutError(f"Timed out after {timeout:.1f}s calling {url}") from exc
    except urllib.error.URLError as exc:
        # On some platforms urllib surfaces a socket timeout wrapped as
        # URLError(reason=socket.timeout()) instead of raising
        # socket.timeout directly — check both.
        if isinstance(exc.reason, socket.timeout):
            raise ProviderTimeoutError(f"Timed out after {timeout:.1f}s calling {url}") from exc
        raise ProviderUnavailableError(f"Connection error calling {url}: {exc.reason}") from exc

    parsed = _safe_json(raw)
    if parsed is None:
        raise ProviderInvalidResponseError(f"Non-JSON response body from {url}")
    return status, parsed
