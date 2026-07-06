"""STUB — Phase 1 (Section 0, step 5).

Compact, injection-safe prompt templates. Every template defined here
MUST wrap any untrusted/external text via agent.safety.wrap_untrusted()
before interpolating it — see Section 9 (REG-07).

No templates are defined yet because the task's actual prompt content
is, by definition, task-specific. Once the task is known, prompts
belong here rather than inline in planner.py, so the whole team can
find and review every string sent to a provider in one place.

Example of the pattern Phase 1 templates should follow (intentionally
not wired into anything yet):

    from agent.safety import wrap_untrusted

    SYSTEM_PROMPT = "..."

    def build_decision_prompt(state_summary: str, external_input: str) -> str:
        return (
            f"{SYSTEM_PROMPT}\\n\\n"
            f"Recent history:\\n{state_summary}\\n\\n"
            f"{wrap_untrusted(external_input, label='OPPONENT_MESSAGE')}"
        )
"""
