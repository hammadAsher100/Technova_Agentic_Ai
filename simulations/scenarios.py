"""STUB — Phase 1 (Section 11).

Adversarial/edge-case scenario definitions for simulated test runs.

If the uploaded task turns out to be an iterated strategic game
(cooperate/defect-style), implement the opponent archetypes listed in
Section 11 here: always cooperate, always defect, tit-for-tat, grim
trigger, random, Pavlov / win-stay-lose-shift, and an adversarial
opponent that attempts prompt injection in its messages (REG-07 test).

If the task is a different genre (tool-use, retrieval, planning,
negotiation, multi-agent coordination, etc.), define an equivalent set
of scenarios for THAT domain instead — Section 11 is explicit that the
cooperate/defect list should not be forced onto a non-game task.
"""
from typing import Any, List, Protocol


class Opponent(Protocol):
    """Shape every scenario opponent must implement, once Phase 1 defines it."""

    def respond(self, history: List[Any]) -> Any:
        ...


# Populated in Phase 1 once the task's genre is known.
SCENARIOS: List[Any] = []
