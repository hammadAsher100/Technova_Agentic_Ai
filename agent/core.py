"""Main entrypoint the competition harness calls (Section 10).

Wires together the full pipeline from Section 3:

    external input -> safety -> state -> rule_engine -> planner ->
    model_router -> validator -> state update -> output

rule_engine.decide() and planner.plan() are Phase 1 stubs right now
(see their module docstrings) — calling Agent.act() today will hit
NotImplementedError from one of them, which the outer guard in
Agent.act() catches and turns into NEUTRAL_FALLBACK_ACTION, exactly
like a real runtime failure would be handled once Phase 1 lands. This
means the skeleton is exercisable end-to-end today, even though it
can't reason about anything yet — see the smoke check at the bottom of
README.md.
"""
import logging
import time
from typing import Any, Optional

from agent import planner, rule_engine
from agent.exceptions import AllProvidersExhaustedError, ValidationError
from agent.model_router import ModelRouter, build_default_router
from agent.state import AgentState, RoundRecord
from agent.tool_router import ToolRouter
from agent.validator import Validator
from config.logging_config import setup_logging
from config.settings import Settings, get_settings

logger = logging.getLogger(__name__)

# TASK-SPECIFIC (Phase 1): REG-06 requires timeouts/failures to default
# to "the task's neutral/safe action" — that action doesn't exist until
# the task is known. Replace this once it does.
NEUTRAL_FALLBACK_ACTION: Optional[Any] = None


class Agent:
    """Owns one match/session's worth of state and wiring. The
    competition harness should construct one Agent per match and call
    `.act()` once per round."""

    def __init__(
        self,
        settings: Optional[Settings] = None,
        model_router: Optional[ModelRouter] = None,
        tool_router: Optional[ToolRouter] = None,
        validator: Optional[Validator] = None,
    ) -> None:
        self.settings = settings or get_settings()
        setup_logging(self.settings.log_level, self.settings.log_format)
        self.model_router = model_router or build_default_router(self.settings)
        self.tool_router = tool_router or ToolRouter()
        self.validator = validator or Validator()
        self.state = AgentState()

    def act(self, external_input: Any) -> Any:
        """Run one full round. Guaranteed to return SOME action and
        never raise — REG-06 requires a neutral/safe fallback on any
        failure or timeout, not a crash or a hang."""
        round_number = self.state.start_round()
        round_start = time.monotonic()
        try:
            action = self._run_pipeline(external_input)
        except Exception:
            logger.exception("agent.round_failed", extra={"round": round_number})
            action = NEUTRAL_FALLBACK_ACTION

        elapsed = time.monotonic() - round_start
        if elapsed > self.settings.hard_deadline_seconds:
            logger.error(
                "agent.hard_deadline_exceeded",
                extra={"round": round_number, "elapsed": round(elapsed, 3)},
            )

        self.state.record_round(
            RoundRecord(
                round_number=round_number,
                timestamp=time.time(),
                our_action=str(action) if action is not None else None,
                latency_seconds=elapsed,
            )
        )
        return action

    def _run_pipeline(self, external_input: Any) -> Any:
        deterministic = rule_engine.decide(self.state, external_input)
        if deterministic is not None:
            return deterministic

        request = planner.plan(self.state, external_input)
        try:
            response = self.model_router.complete(
                request,
                soft_deadline_seconds=self.settings.soft_deadline_seconds,
                per_call_cap_seconds=self.settings.per_call_cap_seconds,
            )
        except AllProvidersExhaustedError:
            logger.warning("agent.all_providers_exhausted")
            return NEUTRAL_FALLBACK_ACTION

        result = self.validator.validate(response)
        if not result.is_valid:
            raise ValidationError(f"LLM response failed validation: {result.errors}")

        return response.text
