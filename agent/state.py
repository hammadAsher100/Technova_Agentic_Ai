"""Own-implementation memory/state tracking (Section 6).

Generic base (Phase 0): round counting, raw history, and compact
summarization. TASK-SPECIFIC EXTENSION POINT (Phase 1): once the task
is known, replace ad-hoc use of `RoundRecord.extra` / `AgentState.extra`
below with properly typed fields — e.g. detected opponent archetype for
an iterated game, or whatever the task's scoring actually depends on
(see Section 6 and Section 11). `extra` is a deliberately loose escape
hatch until then, not a place to leave permanent state.
"""
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RoundRecord:
    round_number: int
    timestamp: float
    our_action: Optional[str] = None
    opponent_action: Optional[str] = None
    reasoning_summary: Optional[str] = None
    provider_used: Optional[str] = None
    latency_seconds: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentState:
    started_at: float = field(default_factory=time.monotonic)
    round_number: int = 0
    history: List[RoundRecord] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)

    def start_round(self) -> int:
        self.round_number += 1
        return self.round_number

    def record_round(self, record: RoundRecord) -> None:
        self.history.append(record)

    def recent_history(self, n: int = 10) -> List[RoundRecord]:
        return self.history[-n:]

    def summarize(self, max_rounds: int = 10) -> str:
        """Compact textual summary for LLM prompts (Section 6: "never
        dump raw full history into a prompt"). Phase 1 should tailor
        this to whatever the task's scoring actually cares about; this
        default just lists recent action pairs."""
        recent = self.recent_history(max_rounds)
        if not recent:
            return "No rounds played yet."
        lines = [
            f"R{r.round_number}: us={r.our_action!r} opponent={r.opponent_action!r}"
            for r in recent
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "round_number": self.round_number,
            "history": [
                {
                    "round_number": r.round_number,
                    "timestamp": r.timestamp,
                    "our_action": r.our_action,
                    "opponent_action": r.opponent_action,
                    "reasoning_summary": r.reasoning_summary,
                    "provider_used": r.provider_used,
                    "latency_seconds": r.latency_seconds,
                    "extra": r.extra,
                }
                for r in self.history
            ],
            "extra": self.extra,
        }
