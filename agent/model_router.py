"""Model Router — provider/model selection, fallback chain, and
per-provider health & quota tracking (Sections 5 and 8).

Deliberately generic (Section 0, step 4): this module does NOT decide
*whether* an LLM call is needed at all — that decision belongs to
planner.py once the task is known. This module only decides, given
that a call must happen, *which* provider+model handles it, in what
order, and how the fallback chain behaves under failure, timeout, and
quota pressure.

Retry strategy note: there is no generic retry decorator (e.g.
tenacity) here on purpose, and not only because tenacity currently
requires Python >=3.10 (see docs/ARCHITECTURE.md). A generic per-call
retry decorator has no visibility into a *shared* deadline across a
multi-provider fallback chain — it can retry one provider's call, but
it can't know "we have 4.2s left across however many providers
remain." `ModelRouter.complete()` tracks one deadline across the whole
chain instead, which is what Section 5 actually asks for ("maintain a
soft internal timeout... abort remaining attempts").
"""
import logging
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

from agent.exceptions import (
    AllProvidersExhaustedError,
    ProviderAuthError,
    ProviderError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from agent.providers.base import BaseProviderClient, ComplexityTier, LLMRequest, LLMResponse
from config.settings import Settings

logger = logging.getLogger(__name__)

_LATENCY_WINDOW = 20  # rolling window size for health scoring
_COOLDOWN_BASE_SECONDS = 5.0
_COOLDOWN_MAX_SECONDS = 60.0
_MIN_VIABLE_CALL_SECONDS = 1.5  # don't start a new attempt with less budget than this
_MINUTE_WINDOW_SECONDS = 60.0
_DAY_WINDOW_SECONDS = 86400.0


@dataclass
class ProviderProfile:
    profile_id: str
    provider_key: str  # groups related profiles for quota purposes, e.g. gemini_flash & gemini_pro -> "gemini"
    client: BaseProviderClient
    model: str
    base_priority: int  # lower = tried first, all else equal
    min_complexity: ComplexityTier = ComplexityTier.LOW
    max_complexity: ComplexityTier = ComplexityTier.HIGH
    per_minute_quota: Optional[int] = None
    per_day_quota: Optional[int] = None
    timeout_cap_seconds: Optional[float] = None

    def suitable_for(self, complexity: ComplexityTier) -> bool:
        return self.min_complexity.value <= complexity.value <= self.max_complexity.value


@dataclass
class _ProviderHealth:
    consecutive_failures: int = 0
    total_calls: int = 0
    total_errors: int = 0
    recent_latencies: Deque[float] = field(default_factory=lambda: deque(maxlen=_LATENCY_WINDOW))
    cooldown_until: float = 0.0
    minute_window_start: float = field(default_factory=time.monotonic)
    minute_call_count: int = 0
    day_window_start: float = field(default_factory=time.monotonic)
    day_call_count: int = 0

    def record_success(self, latency: float) -> None:
        self.consecutive_failures = 0
        self.total_calls += 1
        self.recent_latencies.append(latency)

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_calls += 1
        self.total_errors += 1

    def average_latency(self) -> Optional[float]:
        if not self.recent_latencies:
            return None
        return sum(self.recent_latencies) / len(self.recent_latencies)

    def is_cooling_down(self, now: float) -> bool:
        return now < self.cooldown_until

    def enter_cooldown(self, now: float) -> None:
        """Section 8: 'temporarily deprioritize, not permanently
        blacklist' — exponential backoff capped at _COOLDOWN_MAX_SECONDS,
        not an unbounded or permanent exclusion."""
        backoff = min(
            _COOLDOWN_BASE_SECONDS * (2 ** max(0, self.consecutive_failures - 1)),
            _COOLDOWN_MAX_SECONDS,
        )
        self.cooldown_until = now + backoff

    def register_call_attempt(self, now: float) -> None:
        if now - self.minute_window_start >= _MINUTE_WINDOW_SECONDS:
            self.minute_window_start = now
            self.minute_call_count = 0
        self.minute_call_count += 1

        if now - self.day_window_start >= _DAY_WINDOW_SECONDS:
            self.day_window_start = now
            self.day_call_count = 0
        self.day_call_count += 1


class ModelRouter:
    def __init__(self, profiles: List[ProviderProfile]) -> None:
        if not profiles:
            raise ValueError("ModelRouter requires at least one ProviderProfile")
        self._profiles = profiles
        self._health: Dict[str, _ProviderHealth] = {p.profile_id: _ProviderHealth() for p in profiles}
        self._provider_quota: Dict[str, _ProviderHealth] = {}
        for profile in profiles:
            self._provider_quota.setdefault(profile.provider_key, _ProviderHealth())

    def complete(
        self,
        request: LLMRequest,
        soft_deadline_seconds: float,
        per_call_cap_seconds: float = 10.0,
    ) -> LLMResponse:
        """Try candidates in ranked order until one succeeds or the soft
        deadline is reached.

        Raises AllProvidersExhaustedError if every candidate fails or
        there's no time budget left — the caller (planner.py, Phase 1)
        owns the local heuristic fallback per Section 5, not this
        method.
        """
        deadline = time.monotonic() + soft_deadline_seconds
        candidates = self._ranked_candidates(request)
        last_error: Optional[Exception] = None

        for profile in candidates:
            remaining = deadline - time.monotonic()
            if remaining < _MIN_VIABLE_CALL_SECONDS:
                logger.warning(
                    "model_router.soft_deadline_reached",
                    extra={"remaining_seconds": round(remaining, 3)},
                )
                break

            call_timeout = min(remaining, profile.timeout_cap_seconds or per_call_cap_seconds)
            health = self._health[profile.profile_id]
            health.register_call_attempt(time.monotonic())
            self._provider_quota[profile.provider_key].register_call_attempt(time.monotonic())

            try:
                response = profile.client.complete(request, timeout=call_timeout)
            except ProviderAuthError as exc:
                # Not retryable and not a latency/health signal — a
                # misconfigured key needs a human. Log loudly and move on.
                logger.error("model_router.auth_error", extra={"provider": profile.profile_id})
                last_error = exc
                continue
            except ProviderRateLimitError as exc:
                health.record_failure()
                health.enter_cooldown(time.monotonic())
                logger.warning("model_router.rate_limited", extra={"provider": profile.profile_id})
                last_error = exc
                continue
            except ProviderTimeoutError as exc:
                health.record_failure()
                if health.consecutive_failures >= 2:
                    health.enter_cooldown(time.monotonic())
                logger.warning(
                    "model_router.timeout",
                    extra={"provider": profile.profile_id, "timeout": call_timeout},
                )
                last_error = exc
                continue
            except ProviderError as exc:
                health.record_failure()
                if health.consecutive_failures >= 2:
                    health.enter_cooldown(time.monotonic())
                logger.warning(
                    "model_router.provider_error",
                    extra={"provider": profile.profile_id, "error": str(exc)},
                )
                last_error = exc
                continue

            health.record_success(response.latency_seconds)
            logger.info(
                "model_router.success",
                extra={"provider": profile.profile_id, "latency": round(response.latency_seconds, 3)},
            )
            return response

        raise AllProvidersExhaustedError(
            f"No provider produced a response within the soft deadline "
            f"(last error: {last_error!r})"
        )

    def _ranked_candidates(self, request: LLMRequest) -> List[ProviderProfile]:
        now = time.monotonic()
        # Graceful, three-level widening so we never return an empty
        # pool: prefer quota-ok + complexity-suitable, fall back to
        # quota-ok of any complexity, fall back to everything (letting
        # the provider's own 429 be the final word if our local quota
        # estimate is stale).
        quota_ok = [p for p in self._profiles if self._has_quota_headroom(p, now)]
        by_complexity = [p for p in quota_ok if p.suitable_for(request.complexity)]
        pool = by_complexity or quota_ok or list(self._profiles)
        return sorted(pool, key=lambda p: self._sort_key(p, now))

    def _has_quota_headroom(self, profile: ProviderProfile, now: float) -> bool:
        health = self._provider_quota[profile.provider_key]
        if profile.per_minute_quota is not None:
            window_age = now - health.minute_window_start
            count = 0 if window_age >= _MINUTE_WINDOW_SECONDS else health.minute_call_count
            if count >= profile.per_minute_quota:
                return False
        if profile.per_day_quota is not None:
            window_age = now - health.day_window_start
            count = 0 if window_age >= _DAY_WINDOW_SECONDS else health.day_call_count
            if count >= profile.per_day_quota:
                return False
        return True

    def _sort_key(self, profile: ProviderProfile, now: float) -> Tuple[int, float, int]:
        health = self._health[profile.profile_id]
        cooling = 1 if health.is_cooling_down(now) else 0
        avg_latency = health.average_latency()
        latency_score = avg_latency if avg_latency is not None else 0.0
        return (cooling, float(profile.base_priority), int(latency_score * 1000))

    def health_snapshot(self) -> Dict[str, Dict[str, object]]:
        """For logging/debugging/audit — not consulted by the routing
        decision itself (that reads self._health directly)."""
        now = time.monotonic()
        return {
            profile_id: {
                "consecutive_failures": h.consecutive_failures,
                "total_calls": h.total_calls,
                "total_errors": h.total_errors,
                "average_latency_seconds": h.average_latency(),
                "cooling_down": h.is_cooling_down(now),
            }
            for profile_id, h in self._health.items()
        }


def build_default_router(settings: Settings) -> "ModelRouter":
    """Construct the Section 5 default provider chain from Settings,
    skipping any provider without a configured API key.

    Imports of concrete provider clients are local to this function so
    that constructing a ModelRouter directly with fake profiles (see
    tests/test_model_router_fallback.py) never has to import any
    concrete provider client module at all.
    """
    from agent.providers.gemini_client import GeminiClient
    from agent.providers.groq_client import GroqClient
    profiles: List[ProviderProfile] = []

    order = [settings.primary_llm_provider, settings.fallback_llm_provider]
    if "groq" in order and settings.groq_api_key:
        profiles.append(
            ProviderProfile(
                profile_id="groq",
                provider_key="groq",
                client=GroqClient(api_key=settings.groq_api_key, model=settings.groq_model),
                model=settings.groq_model,
                base_priority=order.index("groq") * 10,
                per_minute_quota=settings.quota_ceilings["groq"].per_minute,
                per_day_quota=settings.quota_ceilings["groq"].per_day,
                timeout_cap_seconds=settings.groq_timeout_seconds,
            )
        )
    else:
        logger.info("model_router.provider_skipped_no_key", extra={"provider": "groq"})

    if "gemini" in order and settings.gemini_api_key:
        profiles.append(
            ProviderProfile(
                profile_id="gemini_flash",
                provider_key="gemini",
                client=GeminiClient(
                    api_key=settings.gemini_api_key,
                    model=settings.gemini_model,
                    provider_id="gemini",
                ),
                model=settings.gemini_model,
                base_priority=order.index("gemini") * 10,
                per_minute_quota=settings.quota_ceilings["gemini"].per_minute,
                per_day_quota=settings.quota_ceilings["gemini"].per_day,
                timeout_cap_seconds=settings.gemini_timeout_seconds,
            )
        )
    else:
        logger.info("model_router.provider_skipped_no_key", extra={"provider": "gemini"})

    if not profiles:
        raise ValueError(
            "No provider API keys configured. Set at least one of "
            "GROQ_API_KEY, GEMINI_API_KEY, MISTRAL_API_KEY, "
            "OPENROUTER_API_KEY in .env (see .env.example)."
        )
    return ModelRouter(profiles)
