# Architecture

> **Phase 1 TODO:** expand "Rule Engine", "Planner", "Memory Fields", and
> "Simulated Opponents" once the competition task is uploaded (Section 12,
> Phase 1 item 11). Everything else below is accurate as of Phase 0.

## Pipeline (Section 3)

```
external input
  -> agent/safety.py         (wrap/flag untrusted text)
  -> agent/state.py           (own memory/history)
  -> agent/rule_engine.py      (deterministic, zero-API — Phase 1 stub)
  -> agent/planner.py           (decide if/what needs an LLM — Phase 1 stub)
  -> agent/tool_router.py        (non-LLM capabilities — Phase 1, if needed)
  -> agent/model_router.py        (which provider, fallback chain)
  -> agent/providers/*_client.py   (transport-only HTTP call)
  -> agent/validator.py             (sanity-check before use)
  -> agent/state.py                  (record the round)
  -> output
```

`agent/core.py::Agent.act()` wires all of this and guarantees a return
value under every failure mode (REG-06) — right now that means "a
`NotImplementedError` from a Phase 1 stub gets caught and turned into
`NEUTRAL_FALLBACK_ACTION`", which becomes real deterministic/LLM
behavior once Phase 1 lands.

## Dependency strategy — why there are no runtime pip packages

REG-02: the agent must run on Python 3.9+, and the judging environment
may only have 3.9. During Phase 0 planning, every dependency the
original spec named was checked against live PyPI metadata. Confidence
varies by package — stated honestly below rather than smoothed over:

| Package | Requires-Python | Confirmed via |
|---|---|---|
| `mistralai` (v2.x) | `>=3.10` | PyPI page + SDK docs, directly, multiple times (v1.x was still on an older floor — Mistral's own PyPI notes describe a "grace period" before this bump) |
| `tenacity` | `>=3.10` | PyPI page, directly (latest checked: 9.1.4) |
| `requests` | `>=3.10` | PyPI page, directly (latest checked: 2.34.2), corroborated by its own changelog: "Dropped support for Python 3.9 following its end of support" |
| `groq` | `>=3.10` | Official GitHub README + PyPI description state this floor explicitly; a third-party citation dated Apr 2026 put the latest release at v1.2.0, but that specific number wasn't independently confirmed against PyPI directly |
| `google-genai` | Not cleanly confirmed either way | Some cached PyPI metadata still shows `>=3.9`. However, its own listed runtime dependencies include `requests` and `tenacity` — both confirmed `>=3.10` above — so even a `google-genai` release that still *declares* a 3.9 floor cannot resolve a working dependency set on a 3.9 interpreter today. Treated as 3.10+-only on that basis. |

Every package checked is 3.10+-only in practice, whether by its own
declared floor or transitively through its dependencies. This reads as
an ecosystem-wide pattern following Python 3.9's October 2025
end-of-life, not a one-off — but it was checked package by package
during Phase 0 rather than assumed, and the table above says exactly
what was and wasn't directly verifiable. Two options existed:

1. **Pin old, pre-drop versions.** Rejected: doing this correctly means
   knowing the exact last 3.9-compatible release per package, which
   isn't something this environment could verify with confidence, and
   it leaves the team depending on unmaintained branches during a live
   competition.
2. **Stdlib-only runtime.** Chosen. `urllib.request` covers HTTP, a
   ~15-line parser in `config/settings.py` covers `.env` loading, and
   `agent/model_router.py` implements its own deadline-aware retry
   logic.

### This is a design improvement, not just a compliance workaround

A generic per-call retry decorator (tenacity-style) has no visibility
into a *shared* deadline across a multi-provider fallback chain — it
can retry one provider's call, but it can't know "we have 4.2s left
across however many providers remain." `ModelRouter.complete()` tracks
one deadline across the whole chain instead, which is the actual
behavior Section 5 asks for ("maintain a soft internal timeout... abort
remaining attempts").

### Upgrade path if Python 3.10+ is confirmed safe

Every provider client in `agent/providers/` implements the same
`BaseProviderClient.complete(request, timeout) -> LLMResponse`
interface (`agent/providers/base.py`). Swapping any one provider from
raw HTTP to its official SDK is a self-contained change to that one
file — nothing in `model_router.py`, `validator.py`, or anywhere else
needs to change. Each affected client's module docstring says so
explicitly.

## Provider REST contracts implemented

- **Groq, Mistral, OpenRouter** — OpenAI-compatible `/chat/completions`
  (shared mapping in `agent/providers/_openai_compatible.py`)
- **Gemini** — native `generateContent` REST contract
  (`agent/providers/gemini_client.py`), authenticated via the
  `x-goog-api-key` header

## Provider health, cooldown, and quota (Sections 5 & 8)

`ModelRouter` ranks candidates on every call by:

1. **Quota headroom** — providers with a configured `per_minute_quota` /
   `per_day_quota` that's currently exhausted are excluded from
   consideration entirely (a 429 is pointless to attempt).
2. **Complexity fit** — `ProviderProfile.min_complexity` /
   `max_complexity` restrict which providers are considered for a given
   `ComplexityTier`; Gemini Pro is scoped to `HIGH` only, to protect its
   tighter quota, exactly as Section 5 specifies.
3. **Cooldown** — a provider that has failed enters a temporary,
   exponentially-backed-off cooldown (capped at 60s) and is
   *deprioritized*, not excluded — if every other candidate is also
   unavailable, a cooling-down provider still gets tried rather than
   giving up outright.
4. **Recent average latency, then configured base priority** — ties are
   broken by the fixed Section 5 ordering (Groq → Gemini Flash → Gemini
   Pro → Mistral → OpenRouter).

All four of these degrade gracefully: if applying every filter leaves
zero candidates, the pool progressively widens (see
`ModelRouter._ranked_candidates`) rather than raising — the actual
provider is always the final word via a real HTTP error, not a locally
stale quota estimate.

## Rule Engine

_Phase 1 TODO — depends on the uploaded task._

## Planner

_Phase 1 TODO — depends on the uploaded task._

## Memory Fields

_Phase 1 TODO — depends on the uploaded task. Generic base: `agent/state.py`._

## Simulated Opponents / Scenarios

_Phase 1 TODO — depends on the uploaded task's genre (Section 11)._
