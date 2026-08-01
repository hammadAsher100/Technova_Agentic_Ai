from agent.state import Move
from simulations.scenarios import SCENARIOS, play

def test_all_scenarios_are_deterministic_and_legal():
    for opponent in SCENARIOS.values():
        assert play(lambda h: Move.COOPERATE, opponent) == play(lambda h: Move.COOPERATE, opponent)

def test_known_targets():
    assert play(lambda h: Move.DEFECT, SCENARIOS["pacifist"]) == 35
    assert play(lambda h: Move.DEFECT, SCENARIOS["predator"]) == 7
