"""Provider-failure and soft-timeout fallback tests for ModelRouter.

Written now (ahead of the Section 12 "Phase 1" test-writing step)
because model_router.py is fully implemented, task-agnostic Phase 0
code (Section 0, step 4) — there's no reason to leave core reliability
behavior unverified until the task arrives. test_rule_engine.py and
test_task_simulations.py remain stubs because THOSE genuinely can't be
written without the task.
"""
import time
from typing import List, Union

import pytest

from agent.exceptions import (
    AllProvidersExhaustedError,
    ProviderAuthError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    ProviderUnavailableError,
)
from agent.model_router import ModelRouter, ProviderProfile
from agent.providers.base import (
    BaseProviderClient,
    ChatMessage,
    ComplexityTier,
    LLMRequest,
    LLMResponse,
)


class _ScriptedClient(BaseProviderClient):
    """Test double: replays a scripted sequence of outcomes per call,
    optionally sleeping first to simulate a slow provider."""

    def __init__(
        self,
        provider_id: str,
        outcomes: List[Union[Exception, LLMResponse]],
        delay_seconds: float = 0.0,
    ) -> None:
        self.provider_id = provider_id
        self._outcomes = list(outcomes)
        self._delay_seconds = delay_seconds
        self.call_count = 0

    def complete(self, request: LLMRequest, timeout: float) -> LLMResponse:
        self.call_count += 1
        if self._delay_seconds:
            time.sleep(self._delay_seconds)
        if not self._outcomes:
            raise AssertionError(f"{self.provider_id} called more times than scripted")
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def _request(complexity: ComplexityTier = ComplexityTier.MEDIUM) -> LLMRequest:
    return LLMRequest(messages=[ChatMessage(role="user", content="hello")], complexity=complexity)


def _response(provider: str, text: str = "ok") -> LLMResponse:
    return LLMResponse(text=text, provider=provider, model="test-model", latency_seconds=0.01)


def _profile(profile_id: str, client: BaseProviderClient, priority: int) -> ProviderProfile:
    return ProviderProfile(
        profile_id=profile_id, provider_key=profile_id, client=client, model="test-model", base_priority=priority
    )


def test_falls_through_to_second_provider_on_failure() -> None:
    primary = _ScriptedClient("primary", [ProviderUnavailableError("boom")])
    backup = _ScriptedClient("backup", [_response("backup")])
    router = ModelRouter([_profile("primary", primary, 10), _profile("backup", backup, 20)])

    response = router.complete(_request(), soft_deadline_seconds=5.0)

    assert response.provider == "backup"
    assert primary.call_count == 1
    assert backup.call_count == 1


def test_all_providers_exhausted_raises() -> None:
    a = _ScriptedClient("a", [ProviderUnavailableError("down")])
    b = _ScriptedClient("b", [ProviderTimeoutError("slow")])
    router = ModelRouter([_profile("a", a, 10), _profile("b", b, 20)])

    with pytest.raises(AllProvidersExhaustedError):
        router.complete(_request(), soft_deadline_seconds=5.0)


def test_soft_deadline_stops_before_exhausting_all_candidates() -> None:
    """Section 5: soft internal timeout must abort remaining attempts
    rather than exhaust every candidate no matter how long that takes."""
    always_times_out = _ScriptedClient(
        "always_times_out", [ProviderTimeoutError("slow")] * 10, delay_seconds=0.3
    )
    router = ModelRouter([_profile("always_times_out", always_times_out, 10)])

    start = time.monotonic()
    with pytest.raises(AllProvidersExhaustedError):
        router.complete(_request(), soft_deadline_seconds=2.2, per_call_cap_seconds=1.0)
    elapsed = time.monotonic() - start

    assert elapsed < 2.5  # aborted well before 10 * 0.3s of scripted delay would take
    assert 1 <= always_times_out.call_count <= 5


def test_rate_limited_provider_is_deprioritized_next_call() -> None:
    limited = _ScriptedClient("limited", [ProviderRateLimitError("429"), _response("limited")])
    healthy = _ScriptedClient("healthy", [_response("healthy"), _response("healthy")])
    router = ModelRouter([_profile("limited", limited, 10), _profile("healthy", healthy, 20)])

    first = router.complete(_request(), soft_deadline_seconds=5.0)
    assert first.provider == "healthy"

    second = router.complete(_request(), soft_deadline_seconds=5.0)
    assert second.provider == "healthy"
    assert limited.call_count == 1  # still cooling down, never retried


def test_auth_error_is_not_retried_but_does_not_block_fallback() -> None:
    misconfigured = _ScriptedClient("misconfigured", [ProviderAuthError("bad key")])
    backup = _ScriptedClient("backup", [_response("backup")])
    router = ModelRouter([_profile("misconfigured", misconfigured, 10), _profile("backup", backup, 20)])

    response = router.complete(_request(), soft_deadline_seconds=5.0)

    assert response.provider == "backup"


def test_high_complexity_excludes_low_tier_only_profile() -> None:
    workhorse = _ScriptedClient("workhorse", [])
    pro = _ScriptedClient("pro", [_response("pro")])
    router = ModelRouter(
        [
            ProviderProfile(
                profile_id="workhorse",
                provider_key="workhorse",
                client=workhorse,
                model="test-model",
                base_priority=10,
                min_complexity=ComplexityTier.LOW,
                max_complexity=ComplexityTier.MEDIUM,
            ),
            ProviderProfile(
                profile_id="pro",
                provider_key="pro",
                client=pro,
                model="test-model",
                base_priority=20,
                min_complexity=ComplexityTier.HIGH,
                max_complexity=ComplexityTier.HIGH,
            ),
        ]
    )

    response = router.complete(_request(complexity=ComplexityTier.HIGH), soft_deadline_seconds=5.0)

    assert response.provider == "pro"
    assert workhorse.call_count == 0


def test_provider_over_quota_ceiling_is_skipped() -> None:
    limited_quota = _ScriptedClient("limited_quota", [_response("limited_quota")])
    unlimited = _ScriptedClient("unlimited", [_response("unlimited")])
    router = ModelRouter(
        [
            ProviderProfile(
                profile_id="limited_quota",
                provider_key="limited_quota",
                client=limited_quota,
                model="test-model",
                base_priority=10,
                per_minute_quota=1,
            ),
            _profile("unlimited", unlimited, 20),
        ]
    )

    first = router.complete(_request(), soft_deadline_seconds=5.0)
    assert first.provider == "limited_quota"

    second = router.complete(_request(), soft_deadline_seconds=5.0)
    assert second.provider == "unlimited"
    assert limited_quota.call_count == 1
