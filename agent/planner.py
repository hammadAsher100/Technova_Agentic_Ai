"""STUB — Phase 1 (Section 0, step 5; Section 12, Phase 1 item 5).

Decides WHETHER an LLM call is warranted for a decision rule_engine
couldn't resolve deterministically, and if so, builds the LLMRequest
(complexity tier, compact state summary, prompt) to hand to
model_router.complete(). This is explicitly where "the decision of
when to escalate to an LLM at all" belongs (Section 0, step 4) — NOT
in model_router.py, which only decides *which provider* handles a call
once planner.py has already decided one is needed. It is also where
the AllProvidersExhaustedError fallback (Section 5's "local heuristic
guess") belongs once model_router.py signals exhaustion.
"""
from typing import Any

from agent.providers.base import LLMRequest
from agent.state import AgentState


def plan(state: AgentState, external_input: Any) -> LLMRequest:
    """NOT IMPLEMENTED — task-specific logic arrives in Phase 1.

    Reminder for when this is implemented (Section 4): send a compact
    *summary* of relevant state (see AgentState.summarize()) — never
    raw full history — to keep prompts small, fast, and cheap on quota.
    Any external/opponent text folded into the prompt MUST go through
    agent.safety.wrap_untrusted() first (REG-07).
    """
    raise NotImplementedError(
        "planner.plan() is a Phase 1 stub. It requires the uploaded "
        "competition task to know what warrants an LLM call and what the "
        "resulting prompt should contain. See Section 0 / Section 12 of "
        "the master build prompt."
    )
