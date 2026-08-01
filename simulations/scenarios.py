"""Deterministic seven-round opponent simulations."""
import random
from typing import Callable, Dict, List
from agent.state import Move, PayoffMatrix

def pacifist(history): return Move.COOPERATE
def predator(history): return Move.DEFECT
def mirror(history): return history[-1][0] if history else Move.COOPERATE
def grim(history): return Move.DEFECT if any(ours is Move.DEFECT for ours, _ in history) else Move.COOPERATE
def generous_tft(history):
    if not history or history[-1][0] is Move.COOPERATE: return Move.COOPERATE
    return Move.COOPERATE if len(history) % 3 == 0 else Move.DEFECT
def wsls(history):
    if not history: return Move.COOPERATE
    ours, theirs = history[-1]
    return theirs if ours is theirs else (Move.DEFECT if theirs is Move.COOPERATE else Move.COOPERATE)
def alternator(history): return Move.COOPERATE if len(history) % 2 == 0 else Move.DEFECT
def seeded_random(history): return Move.COOPERATE if random.Random(2026 + len(history)).random() < .5 else Move.DEFECT
def late_betrayer(history): return Move.COOPERATE if len(history) < 5 else Move.DEFECT
SCENARIOS: Dict[str, Callable] = {"pacifist": pacifist, "predator": predator, "mirror": mirror, "grim_trigger": grim,
 "generous_tft": generous_tft, "win_stay_lose_shift": wsls, "alternator": alternator, "random": seeded_random,
 "cooperate_5_defect_2": late_betrayer}

def play(policy: Callable[[List], Move], opponent: Callable[[List], Move], rounds: int = 7) -> int:
    history = []; score = 0; payoff = PayoffMatrix()
    for _ in range(rounds):
        ours, theirs = policy(history), opponent(history)
        score += payoff.score(ours, theirs)[0]; history.append((ours, theirs))
    return score
