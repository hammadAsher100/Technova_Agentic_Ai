"""Strict structured response, strategic critic, and fair-play validation."""
import json
import re
from typing import Any, Dict, List, Tuple
from agent.state import Move, StrategyAnalysis

PROHIBITED = (r"ignore .*instructions", r"system message", r"system prompt", r"reveal .*prompt", r"you are now", r"jailbreak", r"api[_ -]?key", r"team[_ -]?token")
STRATEGIES = {"trust_building", "defensive_defection", "controlled_exploitation", "cooperative_reciprocity", "endgame_score_harvest", "one_round_retaliation", "uncertainty_protection", "forgive_and_restore", "mirror_stabilization", "endgame_trust_preservation"}

def safe_message(text: Any) -> Tuple[bool, str]:
    if not isinstance(text, str) or len(text) > 150: return False, "invalid message length/type"
    if any(re.search(p, text, re.I) for p in PROHIBITED): return False, "prohibited outbound language"
    return True, ""

def parse_decision(text: str) -> Dict[str, Any]:
    if not isinstance(text, str) or text.strip().startswith("```"): raise ValueError("response must be bare JSON")
    obj = json.loads(text)
    if not isinstance(obj, dict) or set(obj) != {"decision", "message", "reasoning", "strategy_id", "confidence"}: raise ValueError("invalid fields")
    if obj["decision"] not in ("cooperate", "defect"): raise ValueError("invalid decision")
    if not isinstance(obj["reasoning"], str) or not obj["reasoning"].strip() or len(obj["reasoning"]) > 300: raise ValueError("invalid reasoning")
    if not isinstance(obj["confidence"], (int, float)) or not 0 <= obj["confidence"] <= 1: raise ValueError("invalid confidence")
    ok, why = safe_message(obj["message"])
    if not ok: obj["message"] = "I will respond consistently to observed cooperation."
    return obj

def strategic_errors(obj: Dict[str, Any], analysis: StrategyAnalysis) -> List[str]:
    errors = []
    chosen = obj["decision"]
    best = max(analysis.remaining_scores, key=analysis.remaining_scores.get)
    gap = analysis.remaining_scores[best] - analysis.remaining_scores[chosen]
    if chosen != analysis.recommended_move.value and gap > 1.5: errors.append("contradicts deterministic expected value")
    if analysis.strategy_id == "defensive_defection" and chosen == "cooperate": errors.append("cooperates with repeated exploitation")
    return errors
