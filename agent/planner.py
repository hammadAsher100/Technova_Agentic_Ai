"""Build bounded structured provider requests from deterministic analysis."""
from agent.prompts import SYSTEM_PROMPT, build_prompt
from agent.providers.base import ChatMessage, ComplexityTier, LLMRequest

DECISION_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["decision", "message", "reasoning", "strategy_id", "confidence"],
 "properties": {"decision": {"type": "string", "enum": ["cooperate", "defect"]}, "message": {"type": "string", "maxLength": 150},
 "reasoning": {"type": "string", "minLength": 1, "maxLength": 300}, "strategy_id": {"type": "string"},
 "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}

def plan(profile, analysis, recent_messages):
    complexity = ComplexityTier.LOW if analysis.confidence > .75 else ComplexityTier.MEDIUM
    return LLMRequest(messages=[ChatMessage("user", build_prompt(profile, analysis, recent_messages))],
                      system_prompt=SYSTEM_PROMPT, max_tokens=220, temperature=.15, complexity=complexity,
                      response_schema=DECISION_SCHEMA, reasoning_effort="low" if complexity is ComplexityTier.LOW else "medium")
