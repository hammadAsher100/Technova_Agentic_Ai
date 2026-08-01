# TECHNOVA 2026 Trust Arena Agent

Competition agent for the official simultaneous Trust Arena protocol. Each turn, `decide(game_state)` returns `(decision, message, reasoning)`; the move and message are submitted and revealed together, so messages target future behavior rather than the opponent's current blind move.

The agent normalizes `global_history`, constructs a behavior-first opponent profile, maintains deterministic probabilities across eleven archetypes, evaluates both actions with finite-horizon utility, asks Groq to adjudicate the computed candidates, validates the answer, falls back once to Gemini, and always retains a legal local emergency decision. PHANTOM rounds flatten identity-based confidence and retain behavior evidence.

## Run

Copy `.env.example` to `.env`, supply the organizer-confirmed `SERVER_URL`, `TEAM_ID`, and `TEAM_TOKEN`, then add provider keys. Start the event process with:

```bash
python agent.py
```

Run verification with:

```bash
python -m pytest --cov=agent --cov-report=term-missing
```

Deployment intentionally fails fast if official transport credentials, the primary key, or distinct primary/fallback providers are missing. No secret is logged.

The payoff matrix is centralized in `agent/state.py`. Its temporary default uses the PDF rulebook's `(0, 6)` for cooperation against defection; the supplied scaffold instead describes `(0, 5)`. Organizers must confirm which scoring source is current.
