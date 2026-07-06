# Agent Project — Phase 0 Scaffold

Competition-grade agentic AI system. See `master_build_prompt_final__1_.md`
for the full specification this scaffold implements.

**Status:** Phase 0 complete (task-agnostic scaffold, config, provider
transport, model routing, safety, validator). Waiting on the official
competition task upload before Phase 1 (rule engine, planner, prompts,
task-specific state, simulations) begins — see Section 12 of the spec.

## Quickstart

```bash
cd agent-project
cp .env.example .env
# edit .env: fill in API keys for whichever providers you have

pip install -r requirements.txt   # installs pytest only — see below
pytest                            # runs the Phase-0 test suite
```

## A note on dependencies (read this before adding any)

This project runs on the **Python 3.9 standard library only** for
everything in `agent/` and `config/` — no `requests`, no official
provider SDKs, no `tenacity`. This wasn't the original plan; it's a
direct consequence of REG-02 (Python 3.9+ compatibility, judging
environment may only have 3.9) colliding with a real-world fact
discovered during Phase 0: as of mid-2026, every relevant PyPI package
checked (`groq`, `mistralai`, `tenacity`, and `requests` itself) has
directly confirmed its minimum Python version is now 3.10+, following
Python 3.9's October 2025 end-of-life — and `google-genai`, even where
its own metadata was ambiguous, depends on packages that have. Full
rationale, exactly what was directly confirmed vs. inferred, and the
upgrade path if your team confirms 3.10+ is safe: **`docs/ARCHITECTURE.md`**.

`pytest` is the one exception in `requirements.txt` — it's dev/test
tooling that runs on your machine, not inside the judged agent process,
so REG-02's floor doesn't apply to it.

## Project layout

```
config/           All tunables — timeouts, model names, quota ceilings, logging
agent/            Own implementation: state, routing, providers, safety, validation
agent/providers/  Transport-only clients, one per LLM provider, common interface
tests/            Unit tests (some are Phase 1 stubs — see each file's docstring)
simulations/      Adversarial/edge-case scenarios (Phase 1 stub)
docs/             Architecture write-up for teammates/judges
```

## What's implemented right now (Phase 0)

- `config/settings.py` — env loading, every timeout/quota/model constant
- `agent/providers/*` — working Groq, Gemini, Mistral, and OpenRouter clients
- `agent/model_router.py` — fallback chain, soft-timeout budget, per-provider
  health tracking + cooldown, local quota tracking
- `agent/safety.py` — untrusted-input wrapping, injection pattern flagging,
  outbound leak screening (REG-07)
- `agent/validator.py` — extensible output-validation registry
- `agent/state.py` — round/history tracking with a compact `.summarize()`
  for prompts (generic base; task-specific fields land in Phase 1)
- `agent/tool_router.py` — generic tool registry (empty until Phase 1)
- `agent/core.py` — the full pipeline wired end-to-end; runs today and
  returns a safe fallback, since `rule_engine` / `planner` are still stubs

## What's intentionally NOT implemented yet (Phase 1 — waiting on the task)

- `agent/rule_engine.py`, `agent/planner.py`, `agent/prompts.py` — each
  raises `NotImplementedError` with a docstring explaining why
- Task-specific fields in `agent/state.py` (currently a generic `extra` dict)
- `simulations/scenarios.py`, `tests/test_rule_engine.py`,
  `tests/test_task_simulations.py`
- The task-specific sections of `docs/ARCHITECTURE.md`

## Manual smoke check

```bash
python3 -c "
from config.settings import get_settings
s = get_settings()
print('configured providers:', s.configured_providers())
"
```

This raises a clear `ValueError` from `build_default_router()` if you
try to construct an `Agent` before any `.env` key is set — that's
expected until you fill in `.env`.

## Environment variables

See `.env.example` for the full list with inline documentation:
provider API keys, model overrides, timing (`HARD_DEADLINE_SECONDS`,
`SOFT_DEADLINE_SECONDS`, `PER_CALL_CAP_SECONDS`), optional local quota
ceilings, and logging (`LOG_LEVEL`, `LOG_FORMAT`).
