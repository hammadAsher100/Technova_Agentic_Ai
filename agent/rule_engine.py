"""STUB — Phase 1 (Section 0, step 5; Section 12, Phase 1 item 4).

Deterministic, zero-API decision logic (Section 4: "Prefer, in order:
algorithms, heuristics/lookup tables, cached results, only then an LLM
call"). This is where fixed/tested opening behavior, clearly patterned
situations, and repeat scenarios get resolved WITHOUT touching an LLM.

Cannot be written correctly before the task is known — a deterministic
rule for "what to do" only makes sense once we know what "do" means for
this specific competition task. Do not guess at this; wait for the
task upload (Section 0's explicit instruction).
"""
from typing import Any, Optional

from agent.state import AgentState


def decide(state: AgentState, external_input: Any) -> Optional[Any]:
    """Return a deterministic action if one applies, or None to signal
    that planner.py should decide whether an LLM call is warranted.

    NOT IMPLEMENTED — task-specific logic arrives in Phase 1.
    """
    raise NotImplementedError(
        "rule_engine.decide() is a Phase 1 stub. It requires the uploaded "
        "competition task to know what a 'deterministic decision' even "
        "means here. See Section 0 / Section 12 of the master build prompt."
    )
