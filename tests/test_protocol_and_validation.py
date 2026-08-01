import json
from agent.core import TrustArenaAgent
from agent.state import StrategyAnalysis, Move
from agent.validator import parse_decision

def test_strict_json_and_oversized_message_replaced():
    raw=json.dumps({"decision":"cooperate","message":"x"*151,"reasoning":"evidence","strategy_id":"trust_building","confidence":.7})
    assert len(parse_decision(raw)["message"]) <= 150

def test_markdown_rejected():
    try: parse_decision('```json\n{}\n```')
    except ValueError: pass
    else: raise AssertionError("markdown accepted")

class FailingRouter:
    def complete(self,*a,**k):
        from agent.exceptions import AllProvidersExhaustedError
        raise AllProvidersExhaustedError("timeout")

class Settings:
    team_id="us"; turn_budget_seconds=22; submission_reserve_seconds=5; groq_timeout_seconds=8

def test_fallback_and_idempotency():
    agent=TrustArenaAgent(settings=Settings(),router=FailingRouter())
    state={"round_num":1,"opponent_id":"x","global_history":[]}
    first=agent.decide(state); second=agent.decide(state)
    assert first==second and first[0] in ("cooperate","defect") and first[2]

def test_test_mode_reasoning_nonempty():
    agent=TrustArenaAgent(settings=Settings(),router=FailingRouter())
    result=agent.decide({"test_mode":True,"round_num":1,"opponent_id":"t","global_history":[]})
    assert result==( "cooperate", "", "Test mode mandated cooperation.")
