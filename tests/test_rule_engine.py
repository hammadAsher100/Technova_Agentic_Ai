import pytest
from agent.rule_engine import analyze, build_profile, emergency_decision
from agent.state import Move, NormalizedRound, PayoffMatrix, normalize_history

@pytest.mark.parametrize("ours,theirs,score", [(Move.COOPERATE,Move.COOPERATE,(3,3)),(Move.DEFECT,Move.COOPERATE,(5,0)),(Move.COOPERATE,Move.DEFECT,(0,6)),(Move.DEFECT,Move.DEFECT,(1,1))])
def test_payoffs(ours, theirs, score): assert PayoffMatrix().score(ours, theirs) == score

def rows(their_moves):
    return [NormalizedRound(round_number=i+1, opponent_id="x", our_move=Move.COOPERATE, opponent_move=m) for i,m in enumerate(their_moves)]

def test_unknown_opens_cooperate():
    a=analyze(build_profile([], "x"),1,PayoffMatrix()); assert a.recommended_move is Move.COOPERATE
def test_predator_defended():
    a=analyze(build_profile(rows([Move.DEFECT]*3),"x"),4,PayoffMatrix()); assert a.recommended_move is Move.DEFECT
def test_pacifist_exploited_when_known():
    a=analyze(build_profile(rows([Move.COOPERATE]*4),"x"),5,PayoffMatrix()); assert a.recommended_move is Move.DEFECT
def test_probabilities_normalized():
    p=build_profile(rows([Move.COOPERATE,Move.DEFECT]),"x"); assert sum(p.archetype_probabilities.values()) == pytest.approx(1)
def test_normalizer_malformed_never_crashes():
    normalized,warnings=normalize_history([None,{"round_num":"bad"},{"opponent_move":"defect"}],opponent_id="x"); assert len(normalized)==2 and warnings
def test_emergency_always_legal_reasoning():
    result=emergency_decision(analyze(build_profile([],"x"),1,PayoffMatrix())); assert result[0] in ("cooperate","defect") and result[2]
