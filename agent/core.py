"""Official decide pipeline with deterministic emergency and idempotency."""
import logging
import time
from typing import Any, Dict, Optional, Tuple

from agent import planner
from agent.exceptions import AllProvidersExhaustedError
from agent.model_router import ModelRouter, build_default_router
from agent.rule_engine import analyze, build_profile, emergency_decision
from agent.safety import sanitize_untrusted
from agent.state import PayoffMatrix, normalize_history, turn_fingerprint
from agent.validator import parse_decision, strategic_errors
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)
_CACHE: Dict[str, Tuple[str, str, str]] = {}


class TrustArenaAgent:
    def __init__(self, settings: Optional[Settings] = None, router: Optional[ModelRouter] = None,
                 payoff: Optional[PayoffMatrix] = None) -> None:
        self.settings = settings or get_settings()
        self.router = router or build_default_router(self.settings)
        self.payoff = payoff or PayoffMatrix()

    def decide(self, game_state: Dict[str, Any], deadline: Optional[float] = None) -> Tuple[str, str, str]:
        start = time.monotonic()
        key = turn_fingerprint(game_state)
        if key in _CACHE: return _CACHE[key]
        if game_state.get("test_mode"):
            result = ("cooperate", "", "Test mode mandated cooperation.")
            _CACHE[key] = result; return result
        opponent = str(game_state.get("opponent_id") or "UNKNOWN")
        rounds, warnings = normalize_history(game_state.get("global_history", []), self.settings.team_id, opponent)
        if warnings: logger.warning("history.normalization_warnings", extra={"count": len(warnings)})
        phantom = bool(game_state.get("phantom_flag"))
        profile = build_profile(rounds, opponent, phantom)
        analysis = analyze(profile, int(game_state.get("round_num") or 1), self.payoff, phantom)
        local = emergency_decision(analysis)
        if deadline is None: deadline = start + self.settings.turn_budget_seconds
        remaining = deadline - time.monotonic()
        if remaining <= self.settings.submission_reserve_seconds + 1.0:
            _CACHE[key] = local; return local
        recent = []
        for r in rounds[-3:]:
            if r.opponent_message:
                clean, suspicious = sanitize_untrusted(r.opponent_message)
                recent.append({"text": clean, "suspicious": suspicious})
        request = planner.plan(profile, analysis, recent)
        try:
            response = self.router.complete(request, soft_deadline_seconds=max(1.5, remaining - self.settings.submission_reserve_seconds),
                                            per_call_cap_seconds=self.settings.groq_timeout_seconds)
            obj = parse_decision(response.text)
            errors = strategic_errors(obj, analysis)
            if errors:
                logger.warning("decision.overridden", extra={"reasons": errors, "provider": response.provider})
                result = local
            else:
                result = (obj["decision"], obj["message"], obj["reasoning"])
        except (AllProvidersExhaustedError, ValueError, TypeError) as exc:
            logger.warning("decision.local_fallback", extra={"error_type": type(exc).__name__})
            result = local
        if result[0] not in ("cooperate", "defect") or not result[2].strip(): result = local
        _CACHE[key] = (result[0], result[1][:150], result[2][:300])
        return _CACHE[key]


_default_agent: Optional[TrustArenaAgent] = None
def decide(game_state: Dict[str, Any], deadline: Optional[float] = None) -> Tuple[str, str, str]:
    global _default_agent
    if _default_agent is None: _default_agent = TrustArenaAgent()
    return _default_agent.decide(game_state, deadline)
