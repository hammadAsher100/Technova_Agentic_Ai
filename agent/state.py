"""Typed Trust Arena domain model and tolerant history normalization."""
import hashlib
import json
import unicodedata
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple


class Move(Enum):
    COOPERATE = "cooperate"
    DEFECT = "defect"

    @classmethod
    def parse(cls, value: Any) -> Optional["Move"]:
        if isinstance(value, cls):
            return value
        text = str(value or "").strip().lower()
        aliases = {"c": cls.COOPERATE, "cooperate": cls.COOPERATE,
                   "cooperated": cls.COOPERATE, "d": cls.DEFECT,
                   "defect": cls.DEFECT, "defected": cls.DEFECT}
        return aliases.get(text)


@dataclass(frozen=True)
class PayoffMatrix:
    mutual_cooperate: Tuple[int, int] = (3, 3)
    defect_cooperate: Tuple[int, int] = (5, 0)
    cooperate_defect: Tuple[int, int] = (0, 6)  # rulebook default; scaffold says opponent gets 5
    mutual_defect: Tuple[int, int] = (1, 1)

    def score(self, ours: Move, theirs: Move) -> Tuple[int, int]:
        if ours is Move.COOPERATE and theirs is Move.COOPERATE:
            return self.mutual_cooperate
        if ours is Move.DEFECT and theirs is Move.COOPERATE:
            return self.defect_cooperate
        if ours is Move.COOPERATE and theirs is Move.DEFECT:
            return self.cooperate_defect
        return self.mutual_defect


@dataclass
class NormalizedRound:
    match_id: Optional[str] = None
    round_number: Optional[int] = None
    our_id: Optional[str] = None
    opponent_id: Optional[str] = None
    our_move: Optional[Move] = None
    opponent_move: Optional[Move] = None
    our_message: Optional[str] = None
    opponent_message: Optional[str] = None
    our_score: Optional[float] = None
    opponent_score: Optional[float] = None
    timestamp: Optional[Any] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OpponentProfile:
    opponent_id: str
    observed_rounds: int = 0
    cooperation_rate: float = 0.5
    recent_weighted_cooperation_rate: float = 0.5
    defection_after_our_cooperation: float = 0.5
    defection_after_our_defection: float = 0.5
    retaliation_probability: float = 0.5
    forgiveness_probability: float = 0.5
    cooperation_recovery_probability: float = 0.5
    unprovoked_betrayal_probability: float = 0.5
    endgame_betrayal_tendency: float = 0.5
    message_credibility: float = 0.5
    consecutive_cooperations: int = 0
    consecutive_defections: int = 0
    last_move: Optional[Move] = None
    last_message: Optional[str] = None
    prior_matchups: int = 0
    reputation_relevance: float = 0.5
    archetype_probabilities: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.0


@dataclass
class StrategyAnalysis:
    recommended_move: Move
    alternative_move: Move
    cooperation_probability: float
    immediate_scores: Dict[str, float]
    remaining_scores: Dict[str, float]
    reputation_cost: float
    exploitation_risk: float
    trust_value: float
    remaining_rounds: int
    strategy_id: str
    evidence: List[str]
    confidence: float


@dataclass
class Decision:
    move: Move
    message: str
    reasoning: str
    strategy_id: str
    confidence: float
    provider: str = "local"
    latency_seconds: float = 0.0
    fallback_used: bool = False


def _first(record: Dict[str, Any], names: Tuple[str, ...]) -> Any:
    for name in names:
        if name in record:
            return record[name]
    return None


def _text(value: Any, limit: int = 500) -> Optional[str]:
    if value is None:
        return None
    clean = "".join(ch for ch in unicodedata.normalize("NFKC", str(value)) if ch >= " " or ch in "\n\t")
    return clean[:limit]


def normalize_history(raw_history: Any, our_id: Optional[str] = None,
                      opponent_id: Optional[str] = None) -> Tuple[List[NormalizedRound], List[str]]:
    warnings: List[str] = []
    if not isinstance(raw_history, list):
        return [], ["global_history was not a list"]
    result: List[NormalizedRound] = []
    for index, raw in enumerate(raw_history):
        if not isinstance(raw, dict):
            warnings.append("history[%d] was not an object" % index)
            continue
        try:
            explicit_our = _first(raw, ("our_id", "team_id", "player_id", "self_id")) or our_id
            explicit_opp = _first(raw, ("opponent_id", "opponent", "other_id"))
            p1, p2 = _first(raw, ("player1_id", "player_a_id", "participant_a")), _first(raw, ("player2_id", "player_b_id", "participant_b"))
            if not explicit_opp and opponent_id and opponent_id in (p1, p2):
                explicit_opp = opponent_id
            if not explicit_opp and explicit_our in (p1, p2):
                explicit_opp = p2 if explicit_our == p1 else p1
            ours = _first(raw, ("our_move", "our_action", "my_move", "decision", "team_move"))
            theirs = _first(raw, ("opponent_move", "opponent_action", "their_move", "other_move"))
            if p1 and p2 and explicit_our:
                if explicit_our == p1:
                    ours, theirs = _first(raw, ("player1_move", "player_a_move", "move_a")) or ours, _first(raw, ("player2_move", "player_b_move", "move_b")) or theirs
                elif explicit_our == p2:
                    ours, theirs = _first(raw, ("player2_move", "player_b_move", "move_b")) or ours, _first(raw, ("player1_move", "player_a_move", "move_a")) or theirs
            round_value = _first(raw, ("round_num", "round_number", "round"))
            try: round_number = int(round_value) if round_value is not None else None
            except (TypeError, ValueError): round_number = None; warnings.append("history[%d] invalid round" % index)
            known = {"match_id", "game_id", "round_num", "round_number", "round", "our_id", "team_id", "player_id", "self_id", "opponent_id", "opponent", "other_id", "our_move", "our_action", "my_move", "decision", "team_move", "opponent_move", "opponent_action", "their_move", "other_move", "our_message", "my_message", "message", "opponent_message", "their_message", "our_score", "my_score", "opponent_score", "their_score", "timestamp"}
            result.append(NormalizedRound(
                match_id=_text(_first(raw, ("match_id", "game_id"))), round_number=round_number,
                our_id=_text(explicit_our), opponent_id=_text(explicit_opp), our_move=Move.parse(ours), opponent_move=Move.parse(theirs),
                our_message=_text(_first(raw, ("our_message", "my_message", "message")), 150), opponent_message=_text(_first(raw, ("opponent_message", "their_message")), 150),
                our_score=_first(raw, ("our_score", "my_score")), opponent_score=_first(raw, ("opponent_score", "their_score")), timestamp=raw.get("timestamp"),
                metadata={k: v for k, v in raw.items() if k not in known and not any(s in k.lower() for s in ("token", "secret", "key"))}))
        except Exception:
            warnings.append("history[%d] could not be normalized" % index)
    return result, warnings


def turn_fingerprint(game_state: Dict[str, Any]) -> str:
    safe = {"match_id": game_state.get("match_id") or game_state.get("game_id"), "round_num": game_state.get("round_num"),
            "opponent_id": game_state.get("opponent_id"), "global_history": game_state.get("global_history", [])}
    return hashlib.sha256(json.dumps(safe, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8")).hexdigest()
