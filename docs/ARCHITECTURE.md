# Standalone Competition Architecture

All production logic lives in root `agent.py`; the deleted modular runtime remains available in Git history only. The file is organized as configuration, typed domain model, safety and history normalization, opponent profiling, archetype policies, bounded rollout, strategic invariants, prompt/Groq transport, strict validation and critic, deterministic fallback, instance cache, official arena transport, and main loop.

The live path is:

```text
game state -> normalized current-opponent history -> probabilistic profile
-> both-action rollout through TOTAL_ROUNDS -> invariants -> one Groq call
-> strict parser/fair-play critic -> accepted tuple OR deterministic tuple
```

No provider router, secondary model client, SDK, framework, service, database, or background thread is present. Both Groq and arena HTTPS use `urllib.request` with actual network timeouts. One monotonic deadline preserves submission reserve. Test mode cooperates with nonempty reasoning; practice and live endpoints remain separated.

The payoff discrepancy remains unresolved: rulebook `(0, 6)`, scaffold `(0, 5)` for us cooperating against opponent defection. `PayoffMatrix` centralizes the temporary rulebook default.
