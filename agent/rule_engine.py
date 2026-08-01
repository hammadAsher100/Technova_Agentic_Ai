"""Deterministic opponent modelling and finite-horizon strategy engine."""
from typing import Dict, List, Sequence

from agent.state import Move, NormalizedRound, OpponentProfile, PayoffMatrix, StrategyAnalysis

ARCHETYPES = ("pacifist", "predator", "mirror", "generous_tft", "grim_trigger",
              "win_stay_lose_shift", "alternator", "random", "opportunist",
              "endgame_betrayer", "strategic_unknown")


def _ratio(flags: List[bool], default: float = 0.5) -> float:
    return sum(flags) / float(len(flags)) if flags else default


def _normalize(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.001, value) for value in weights.values())
    return {key: max(0.001, value) / total for key, value in weights.items()}


def build_profile(rounds: Sequence[NormalizedRound], opponent_id: str,
                  phantom: bool = False) -> OpponentProfile:
    relevant = [r for r in rounds if (not r.opponent_id or r.opponent_id == opponent_id)
                and r.opponent_move is not None]
    moves = [r.opponent_move for r in relevant]
    coop = _ratio([m is Move.COOPERATE for m in moves])
    weighted_total = sum(range(1, len(moves) + 1)) or 1
    recent = sum(i for i, m in enumerate(moves, 1) if m is Move.COOPERATE) / float(weighted_total)
    after_c, after_d, retaliation, forgiveness, recovery, betrayals = [], [], [], [], [], []
    for prev, cur in zip(relevant, relevant[1:]):
        if prev.our_move is Move.COOPERATE: after_c.append(cur.opponent_move is Move.DEFECT)
        if prev.our_move is Move.DEFECT: after_d.append(cur.opponent_move is Move.DEFECT)
        if prev.our_move is Move.DEFECT: retaliation.append(cur.opponent_move is Move.DEFECT)
        if prev.opponent_move is Move.DEFECT: forgiveness.append(cur.opponent_move is Move.COOPERATE)
        if prev.opponent_move is Move.DEFECT: recovery.append(cur.opponent_move is Move.COOPERATE)
        if prev.our_move is Move.COOPERATE and prev.opponent_move is Move.COOPERATE:
            betrayals.append(cur.opponent_move is Move.DEFECT)
    credibility_flags: List[bool] = []
    for previous, current in zip(relevant, relevant[1:]):
        message = (previous.opponent_message or "").lower()
        if "cooperat" in message: credibility_flags.append(current.opponent_move is Move.COOPERATE)
        elif "defect" in message: credibility_flags.append(current.opponent_move is Move.DEFECT)
    alternations = _ratio([a is not b for a, b in zip(moves, moves[1:])], 0.0)
    mirror_samples = [cur.opponent_move is prev.our_move for prev, cur in zip(relevant, relevant[1:]) if prev.our_move]
    mirror = _ratio(mirror_samples)
    late = [r.opponent_move is Move.DEFECT for r in relevant if (r.round_number or 0) >= 6]
    endgame = _ratio(late)
    n = len(relevant)
    weights = {name: 0.2 for name in ARCHETYPES}
    weights.update({
        "pacifist": 0.15 + 3.2 * coop,
        "predator": 0.15 + 3.2 * (1 - coop),
        "mirror": 0.2 + 3.0 * mirror,
        "generous_tft": 0.2 + 1.8 * mirror + _ratio(forgiveness),
        "grim_trigger": 0.2 + 2.0 * _ratio(retaliation) + (1.0 - _ratio(forgiveness)),
        "win_stay_lose_shift": 0.2 + 1.2 * mirror,
        "alternator": 0.2 + 3.0 * alternations,
        "random": 0.3 + (1.0 - abs(coop - 0.5) * 2) * (1.0 - abs(alternations - 0.5)),
        "opportunist": 0.2 + 2.0 * _ratio(betrayals),
        "endgame_betrayer": 0.2 + 2.8 * endgame,
        "strategic_unknown": max(0.2, 2.0 - n * 0.25),
    })
    identity = opponent_id.strip().lower().replace("-", "_").replace(" ", "_")
    known_identity = {"pacifist": "pacifist", "predator": "predator", "mirror": "mirror"}.get(identity)
    if known_identity and not phantom:
        weights[known_identity] += 50.0
    if n >= 3 and coop == 1.0: weights["pacifist"] += 5
    if n >= 2 and coop == 0.0: weights["predator"] += 5
    probs = _normalize(weights)
    if phantom:
        probs = _normalize({k: (v * 0.55 + 1.0 / len(probs) * 0.45) for k, v in probs.items()})
    streak_c = streak_d = 0
    for move in reversed(moves):
        if move is Move.COOPERATE and streak_d == 0: streak_c += 1
        elif move is Move.DEFECT and streak_c == 0: streak_d += 1
        else: break
    matchups = len({r.match_id for r in relevant if r.match_id})
    return OpponentProfile(
        opponent_id=opponent_id, observed_rounds=n, cooperation_rate=coop,
        recent_weighted_cooperation_rate=recent, defection_after_our_cooperation=_ratio(after_c),
        defection_after_our_defection=_ratio(after_d), retaliation_probability=_ratio(retaliation),
        forgiveness_probability=_ratio(forgiveness), cooperation_recovery_probability=_ratio(recovery),
        unprovoked_betrayal_probability=_ratio(betrayals), endgame_betrayal_tendency=endgame,
        message_credibility=_ratio(credibility_flags), consecutive_cooperations=streak_c,
        consecutive_defections=streak_d, last_move=moves[-1] if moves else None,
        last_message=relevant[-1].opponent_message if relevant else None, prior_matchups=matchups,
        reputation_relevance=min(1.0, 0.35 + 0.2 * matchups), archetype_probabilities=probs,
        confidence=min(0.95, n / 6.0) * (0.65 if phantom else 1.0))


def analyze(profile: OpponentProfile, round_num: int, payoff: PayoffMatrix,
            phantom: bool = False) -> StrategyAnalysis:
    remaining = max(0, 7 - round_num)
    p = profile.recent_weighted_cooperation_rate if profile.observed_rounds else 0.5
    pred = profile.archetype_probabilities.get("predator", 0)
    pac = profile.archetype_probabilities.get("pacifist", 0)
    mirror = profile.archetype_probabilities.get("mirror", 0) + profile.archetype_probabilities.get("generous_tft", 0)
    exploitation = min(1.0, 0.45 * (1 - p) + 0.35 * pred + 0.2 * profile.unprovoked_betrayal_probability)
    trust = min(1.0, 0.5 * p + 0.25 * mirror + 0.25 * profile.message_credibility)
    c_now = p * payoff.score(Move.COOPERATE, Move.COOPERATE)[0] + (1-p) * payoff.score(Move.COOPERATE, Move.DEFECT)[0]
    d_now = p * payoff.score(Move.DEFECT, Move.COOPERATE)[0] + (1-p) * payoff.score(Move.DEFECT, Move.DEFECT)[0]
    reputation_cost = profile.reputation_relevance * (0.8 + remaining * 0.25)
    retaliation_cost = remaining * mirror * max(0.0, p * 2.0)
    repair_value = remaining * trust * 1.35
    c_total = c_now + remaining * (2.2 + 0.8 * trust) + repair_value * 0.2
    d_total = d_now + remaining * (1.0 + 1.2 * (1 - exploitation)) - retaliation_cost - reputation_cost
    evidence: List[str] = ["%d observed rounds" % profile.observed_rounds,
                           "cooperation rate %.2f" % profile.cooperation_rate]
    if phantom: evidence.append("PHANTOM identity uncertainty reduced reputation confidence")
    identity = profile.opponent_id.strip().lower().replace("-", "_").replace(" ", "_")
    if not phantom and identity in ("pacifist", "predator"):
        recommended, strategy = Move.DEFECT, "controlled_exploitation" if identity == "pacifist" else "defensive_defection"
    elif not phantom and identity == "mirror":
        recommended, strategy = (Move.DEFECT, "endgame_score_harvest") if round_num == 7 and profile.prior_matchups <= 1 else (Move.COOPERATE, "mirror_stabilization")
    elif profile.observed_rounds == 0:
        recommended, strategy = Move.COOPERATE, "trust_building"
    elif pred > 0.42 or profile.consecutive_defections >= 2:
        recommended, strategy = Move.DEFECT, "defensive_defection"
    elif (pac > 0.45 or (profile.observed_rounds >= 3 and profile.cooperation_rate == 1.0)) and profile.confidence >= 0.45:
        recommended, strategy = Move.DEFECT, "controlled_exploitation"
    elif remaining and mirror > 0.30 and trust > 0.50:
        recommended, strategy = Move.COOPERATE, "cooperative_reciprocity"
    elif round_num == 7 and d_total > c_total + reputation_cost:
        recommended, strategy = Move.DEFECT, "endgame_score_harvest"
    elif profile.last_move is Move.DEFECT and profile.consecutive_defections == 1 and trust > 0.55:
        recommended, strategy = Move.DEFECT, "one_round_retaliation"
    else:
        recommended = Move.COOPERATE if c_total >= d_total else Move.DEFECT
        strategy = "uncertainty_protection" if phantom or profile.confidence < 0.4 else "forgive_and_restore"
    alternative = Move.DEFECT if recommended is Move.COOPERATE else Move.COOPERATE
    return StrategyAnalysis(recommended, alternative, p, {"cooperate": c_now, "defect": d_now},
                            {"cooperate": c_total, "defect": d_total}, reputation_cost,
                            exploitation, trust, remaining, strategy, evidence,
                            min(0.95, 0.35 + profile.confidence * 0.6))


def emergency_decision(analysis: StrategyAnalysis) -> tuple:
    move = analysis.recommended_move
    if move is Move.COOPERATE:
        message = "I will reciprocate cooperation and repair isolated mistakes."
    else:
        message = "I respond to observed behavior; reliable cooperation can restore trust."
    reason = "%s: %s" % (analysis.strategy_id.replace("_", " ").title(), "; ".join(analysis.evidence[:2]))
    return move.value, message[:150], reason[:300]
