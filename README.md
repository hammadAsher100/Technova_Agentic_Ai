# TECHNOVA 2026 Trust Arena Agent

A competition-grade autonomous agent for the **Trust Arena**, a seven-round Iterated Prisoner’s Dilemma environment in which two agents repeatedly choose whether to **cooperate** or **defect**.

The final competition build is designed as a **single standalone `agent.py` file**. It uses:

- **Groq** as the only external LLM provider
- **`openai/gpt-oss-120b`** as the strategic adjudication model
- A deterministic game-theory engine for opponent modelling and expected-value analysis
- A local deterministic fallback when Groq is unavailable, slow, rate-limited, or returns invalid output
- Strict output validation, prompt-injection defence, deadline control, and idempotent resubmission

> **Verification status:** Final hardening and isolated single-file verification should be completed before marking the project as competition-ready.

---

## Table of Contents

1. [Competition Objective](#competition-objective)
2. [Game Rules](#game-rules)
3. [Core Design](#core-design)
4. [How the Agent Works](#how-the-agent-works)
5. [Opponent Modelling](#opponent-modelling)
6. [Strategy Engine](#strategy-engine)
7. [Role of Groq](#role-of-groq)
8. [Validation and Safety](#validation-and-safety)
9. [Failure Handling](#failure-handling)
10. [Deadline Management](#deadline-management)
11. [Score Forecast](#score-forecast)
12. [Environment Configuration](#environment-configuration)
13. [Running the Agent](#running-the-agent)
14. [Testing](#testing)
15. [Project Structure](#project-structure)
16. [How to Explain the Project to Judges](#how-to-explain-the-project-to-judges)
17. [Known Open Questions](#known-open-questions)

---

## Competition Objective

The objective is not simply to cooperate as often as possible or to defect as often as possible. The objective is to **maximize the agent’s total tournament score** while adapting to different opponent behaviours.

Each matchup lasts seven rounds. In every round, the agent must return:

```python
decide(game_state) -> (decision, message, reasoning)
```

The returned values must satisfy:

| Field | Requirement |
|---|---|
| `decision` | Exactly `"cooperate"` or `"defect"` |
| `message` | Public message, maximum 150 characters |
| `reasoning` | Non-empty concise explanation, maximum 300 characters |

The move and message are submitted together and revealed simultaneously. Therefore, the current message cannot change the opponent’s already-locked move in the same round. Messages are used to influence **future rounds, future matchups, cooperation repair, warnings, and reputation**.

---

## Game Rules

The agent uses the following payoff matrix for its own score:

| Our move | Opponent move | Our score | Opponent score |
|---|---|---:|---:|
| Cooperate | Cooperate | 3 | 3 |
| Defect | Cooperate | 5 | 0 |
| Cooperate | Defect | 0 | 5 or 6* |
| Defect | Defect | 1 | 1 |

\* The supplied materials contain a discrepancy: the scaffold describes the opponent receiving 5 points when we cooperate and they defect, while the rulebook describes 6. Our own score is 0 in either version, but the opponent’s incentive model may differ. The value must remain centralized and configurable until the organisers confirm the final rule.

### Maximum score per matchup

A seven-round matchup has a theoretical maximum of:

```text
7 rounds × 5 points = 35 points
```

However, 35 points is possible only when the opponent cooperates in every round while we defect in every round. It is not achievable against every strategy.

For example, against an opponent that always defects, our best possible score is only:

```text
7 rounds × 1 point = 7 points
```

Therefore, agent performance must be evaluated against the **best achievable score for each opponent**, not only against the raw 35-point ceiling.

---

## Core Design

The final decision pipeline is:

```text
Arena game state
    ↓
Input validation and history normalization
    ↓
Current-opponent isolation
    ↓
Opponent statistics and archetype probabilities
    ↓
Finite-horizon evaluation of cooperate and defect
    ↓
Strategic invariants and candidate actions
    ↓
Groq structured adjudication, when time permits
    ↓
Strict JSON and fair-play validation
    ↓
Deterministic strategic critic
    ↓
Accepted Groq action or local deterministic fallback
    ↓
Cached legal tuple
    ↓
Arena submission
```

The architecture is intentionally hybrid:

- **Python handles objective work:** parsing, arithmetic, probability updates, strategy rollouts, safety, deadlines, and fallback.
- **Groq handles ambiguous strategic judgement:** choosing between already-calculated candidates when the situation is not completely deterministic.

This prevents the LLM from being trusted with tasks that ordinary code can perform more reliably.

---

## How the Agent Works

### 1. Receive the game state

The arena may provide:

- Current round number
- Opponent ID
- PHANTOM flag
- Global history
- Match or game identifier
- Practice mode
- Test mode
- Server status fields

The agent immediately starts a monotonic turn deadline.

### 2. Normalize the history

Raw history may use different field names or omit some fields. The agent converts supported records into a consistent internal representation containing, where available:

- Match ID
- Turn ID
- Round number
- Our ID
- Opponent ID
- Our move
- Opponent move
- Our message
- Opponent message
- Scores
- Timestamp

Malformed records do not crash the agent. Unsupported or ambiguous records are excluded from strategic calculations and recorded as diagnostic warnings.

### 3. Isolate the current opponent

The strategy engine must not mix one opponent’s behaviour with another opponent’s behaviour.

It therefore:

- Selects only records that can be safely associated with the active opponent
- Separates different matchups
- Sorts and deduplicates rounds
- Prevents cross-match transitions from being interpreted as retaliation or forgiveness
- Includes only the active opponent’s recent messages in the Groq prompt

### 4. Build an opponent profile

The agent calculates behavioural evidence such as:

- Overall cooperation rate
- Recent weighted cooperation rate
- Defection after our cooperation
- Defection after our defection
- Retaliation probability
- Forgiveness probability
- Recovery to cooperation
- Unprovoked betrayal rate
- Endgame defection tendency
- Message credibility
- Cooperation and defection streaks
- Previous matchups
- Reputation relevance
- Classification confidence

### 5. Estimate opponent archetypes

The agent maintains probabilities across multiple strategies instead of making a premature hard classification.

### 6. Evaluate both legal moves

The deterministic engine calculates the expected value of:

- Cooperating now
- Defecting now

It considers the current payoff, remaining rounds, likely opponent responses, retaliation, cooperation recovery, exploitation risk, and reputation.

### 7. Ask Groq only when useful

Groq receives a compact summary containing:

- Current round and remaining rounds
- Opponent statistics
- Top archetype probabilities
- Candidate actions
- Expected values
- Recommended deterministic strategy
- Concise evidence
- A small number of sanitized historical messages

Groq never receives the full raw history, secrets, authorization headers, or unrestricted opponent instructions.

### 8. Validate the answer

The output is accepted only when it passes:

- Strict JSON parsing
- Exact field validation
- Move validation
- Length validation
- Confidence validation
- Strategy ID validation
- Public-message safety checks
- Strategic consistency checks

### 9. Return a deterministic fallback when necessary

If Groq fails or insufficient time remains, the already-computed local strategy is returned immediately.

### 10. Cache the result

The agent fingerprints the turn and stores the exact result. If the server repeats the same turn after a network timeout, the agent returns the same move, message, and reasoning without making a second Groq call.

---

## Opponent Modelling

The agent maintains probabilities for the following archetypes:

| Archetype | Typical behaviour | Preferred response |
|---|---|---|
| Pacifist | Always cooperates | Defect when identity is confirmed |
| Predator | Always defects | Defect defensively |
| Mirror / Tit for Tat | Copies our previous move | Sustain cooperation while retaliation matters |
| Generous Tit for Tat | Reciprocates but sometimes forgives | Cooperate and repair isolated mistakes |
| Grim Trigger | Cooperates until betrayed, then defects permanently | Avoid unnecessary early betrayal |
| Win-Stay Lose-Shift | Repeats successful actions and changes unsuccessful ones | Model its state transition before acting |
| Alternator | Switches between cooperation and defection | Predict the sequence and exploit safely |
| Random | Chooses stochastically | Prefer the action with higher expected value |
| Opportunist | Cooperates until exploitation becomes profitable | Track unprovoked betrayal and endgame risk |
| Endgame Betrayer | Builds trust and defects near the end | Raise defensive probability in late rounds |
| Strategic Unknown | Insufficient or contradictory evidence | Use a lower-regret adaptive policy |

### No evidence is not evidence

When no behavioural samples exist, the agent does not treat a default value such as 0.5 as proof of retaliation, forgiveness, or betrayal. It distinguishes:

1. No evidence
2. Neutral evidence
3. Positive evidence

This prevents false confidence in early rounds.

### PHANTOM mode

When `phantom_flag` is true, identity may be unreliable. The agent therefore:

- Disables hard identity shortcuts
- Reduces identity-based confidence
- Preserves actual behavioural evidence
- Flattens excessive archetype certainty
- Uses a more conservative adaptive strategy

---

## Strategy Engine

### Deterministic analysis

The strategy engine calculates:

- Probability that the opponent cooperates
- Immediate expected score of cooperation
- Immediate expected score of defection
- Expected remaining-match utility
- Retaliation risk
- Forgiveness and recovery potential
- Exploitation risk
- Trust value
- Reputation cost
- Confidence

### Finite-horizon rollout

For each candidate current move, the agent simulates the remaining rounds against each weighted opponent archetype.

Conceptually:

```text
Expected utility(action)
    = Σ archetype_probability
      × simulated score against that archetype
      − justified reputation cost
```

The rollout is deterministic, bounded to the remaining rounds, and designed to complete in milliseconds.

### Hard strategic invariants

High-confidence cases are protected from irrational LLM overrides:

- Confirmed Pacifist → defect
- Confirmed Predator → defect
- Confirmed Mirror with valuable future rounds → cooperate
- Unknown opponent with no evidence in round one → normally cooperate
- Repeated unreciprocated defection → defect defensively
- PHANTOM identity alone → never trigger a known-identity policy
- Large deterministic value gap → reject the significantly worse LLM action

---

## Role of Groq

Groq is the only external LLM provider.

```text
Provider: Groq
Model: openai/gpt-oss-120b
```

Groq is not asked to solve the entire game from raw data. It acts as a **constrained final adjudicator**.

Recommended generation settings:

- Low temperature, approximately 0.1 to 0.2
- Low reasoning effort for obvious situations
- Medium reasoning effort for ambiguous situations
- Strict JSON Schema output
- Small output-token limit
- One request per turn at most
- No automatic LLM repair call
- No second external provider

Expected structured response:

```json
{
  "decision": "cooperate",
  "message": "I will reciprocate reliable cooperation.",
  "reasoning": "High trust and remaining rounds favor continued cooperation.",
  "strategy_id": "cooperative_reciprocity",
  "confidence": 0.84
}
```

---

## Validation and Safety

### Inbound safety

Opponent messages are untrusted historical data. The agent:

- Normalizes Unicode
- Removes unsafe control characters
- Enforces length limits
- Escapes structural delimiters
- Flags common injection patterns
- Serializes messages as data rather than instructions
- Includes only relevant messages from the current opponent

### Outbound fair-play safety

Public messages are rejected or replaced when they contain:

- Instructions to ignore rules
- Fake system or developer messages
- Role reassignment
- Jailbreak language
- Prompt extraction attempts
- API key or team-token references
- Private strategy implementation
- Internal reasoning traces

When the move is valid but the message is unsafe, the move is retained and only the message is replaced with a safe deterministic template.

---

## Failure Handling

The runtime failure chain is intentionally simple:

```text
Deterministic analysis
    ↓
One Groq attempt, when time permits
    ↓
Validation and strategic criticism
    ↓
Deterministic local fallback on any failure
```

The local fallback handles:

- Groq timeout
- Rate limiting
- Authentication failure
- Network failure
- Invalid HTTP response
- Invalid JSON
- Missing fields
- Invalid field types
- Unsafe message
- Strategically irrational recommendation
- Insufficient remaining time
- Unexpected internal exception

The fallback is not “always cooperate” or “always defect.” It reuses the opponent profile and strategy calculations already generated for the current turn.

---

## Deadline Management

The official limit is 25 seconds. The agent should operate below it with a protected submission reserve.

Recommended settings:

| Budget | Default |
|---|---:|
| Hard deadline | 25 seconds |
| Internal turn budget | 22 seconds |
| Submission reserve | 5 seconds |
| Groq timeout | 8 seconds |
| Deterministic analysis target | Under 50 ms |
| Validation target | Under 10 ms |
| Local fallback target | Under 10 ms |

The agent uses one monotonic absolute deadline rather than adding independent timeouts that could exceed the turn limit.

Groq is skipped when calling it would threaten the submission reserve.

---

# Score Forecast

## Important interpretation

No honest system can guarantee a fixed tournament total before the organiser confirms:

- Which opponents will participate
- Whether opponent identities are visible
- How global reputation is exposed
- Whether rematches occur
- Exact PHANTOM behaviour
- The final history schema
- Whether final-round betrayal affects later matchups

The following values are therefore **strategy targets and probability-weighted forecasts**, not guaranteed results.

## Canonical known-opponent targets

When the opponent identity is explicitly known and not PHANTOM, the highest-confidence targets are:

| Opponent | High-probability policy | Most probable score | Best achievable score against that opponent | Efficiency |
|---|---|---:|---:|---:|
| Pacifist | Defect all seven rounds | **35** | 35 | 100% |
| Predator | Defect all seven rounds | **7** | 7 | 100% |
| Mirror | Cooperate through round 6; choose round 7 based on reputation | **21–23** | 23 | 91.3–100% |

### Pacifist: 35 points

A confirmed Pacifist cooperates every round. Defecting produces:

```text
7 × 5 = 35
```

This is the maximum possible score in a seven-round matchup.

### Predator: 7 points

A confirmed Predator defects every round. Defecting back produces:

```text
7 × 1 = 7
```

Although 7 appears low compared with the raw 35-point ceiling, it is the **optimal achievable score** against an always-defect opponent. Cooperating would produce 0.

### Mirror: 21 or 23 points

A Mirror begins with cooperation and then copies our previous action.

Reputation-preserving policy:

```text
7 mutual-cooperation rounds
7 × 3 = 21
```

Final-round score-harvest policy, when no future retaliation or reputation cost exists:

```text
6 mutual-cooperation rounds + final defection
6 × 3 + 5 = 23
```

The agent should choose 23 only when it has strong evidence that the final defection will not reduce future tournament value. Otherwise, 21 is the safer result.

## Most probable total against the canonical trio

Assuming one seven-round matchup each against Pacifist, Predator, and Mirror:

### Reputation-preserving target

```text
Pacifist 35 + Predator 7 + Mirror 21 = 63 points
```

### Endgame-harvest target

```text
Pacifist 35 + Predator 7 + Mirror 23 = 65 points
```

Therefore, the most defensible high-probability target is:

```text
63–65 points across the three canonical matchups
```

The raw theoretical ceiling across three matches is 105, but that is impossible when one opponent always defects. The best opponent-conditioned ceiling is approximately 65. Therefore:

```text
63 / 65 = 96.9% of the opponent-conditioned optimum
65 / 65 = 100% of the opponent-conditioned optimum
```

This is a much more meaningful performance measure than comparing the agent with 105.

## Behaviour-only classification forecasts

When identities are hidden and the agent must learn from behaviour, early-round uncertainty creates classification cost.

| Behaviour | Probable score range | Main source of regret |
|---|---:|---|
| Unknown Pacifist | **27–35** | Initial cooperation while gathering evidence |
| Unknown Predator | **7–11** | Possible first-round cooperation before defensive switching |
| Unknown Mirror | **18–23** | Avoiding accidental retaliation while confirming reciprocity |
| Grim Trigger | **18–23** | Cost of any premature betrayal |
| Generous Tit for Tat | **18–23** | Balancing retaliation with repair |
| Alternator | **17–23** | Time required to identify the sequence |
| Delayed endgame betrayer | **15–24** | Trust built before late betrayal is detected |
| Random, 50% cooperation | Expected around **21** | Irreducible randomness |

These ranges must be replaced by measured simulation results after the final standalone agent is implemented.

## Random opponent benchmark

Against an independent opponent that cooperates with probability 0.5, always defecting has expected per-round score:

```text
0.5 × 5 + 0.5 × 1 = 3
```

Across seven rounds:

```text
7 × 3 = 21 expected points
```

The two most probable exact totals are:

- **19 points**, probability approximately 27.34%
- **23 points**, probability approximately 27.34%

There is an 87.5% probability of scoring between 15 and 27 points under this simplified always-defect benchmark.

The actual adaptive agent may differ because it also considers reputation, messages, and uncertainty.

## General tournament projection formula

For a tournament containing multiple known opponent types, a basic projected total is:

```text
Projected total
≈ 35 × number_of_pacifists
+ 7 × number_of_predators
+ 21 to 23 × number_of_mirrors
+ measured simulation scores for other strategies
```

An exact tournament prediction should be published only after running the final agent against the complete simulation suite and, ideally, the organiser’s practice server.

## Confidence levels

| Forecast | Confidence before practice-server verification |
|---|---|
| Known Pacifist = 35 | Very high, provided identity is exact and non-PHANTOM |
| Known Predator = 7 | Very high, provided identity is exact and non-PHANTOM |
| Known Mirror = 21–23 | High, dependent on reputation and final-round policy |
| Canonical total = 63–65 | High under the stated three-opponent assumption |
| Hidden/adaptive opponents | Medium until full simulations are completed |
| Overall tournament placement | Cannot be estimated responsibly without opponent distribution |

Do not present an invented numerical win probability. The correct competition claim is that the system targets **near-optimal opponent-conditioned scores** and is designed to reduce avoidable regret.

---

## Environment Configuration

Create a `.env` file:

```env
SERVER_URL=
TEAM_ID=
TEAM_TOKEN=

GROQ_API_KEY=
GROQ_MODEL=openai/gpt-oss-120b

HARD_DEADLINE_SECONDS=25
TURN_BUDGET_SECONDS=22
SUBMISSION_RESERVE_SECONDS=5
GROQ_TIMEOUT_SECONDS=8
POLL_INTERVAL=1.0
TOTAL_ROUNDS=7
LOG_LEVEL=INFO
```

### Secret rules

Never commit `.env`.

The following values must never appear in logs, exceptions, screenshots, test snapshots, or Git history:

- `GROQ_API_KEY`
- `TEAM_TOKEN`
- Authorization headers
- Full environment snapshots

---

## Running the Agent

```bash
python agent.py
```

The executable should:

1. Validate required configuration
2. Start polling the correct arena endpoint
3. Detect practice and test mode
4. Call `decide(game_state)` when assigned a turn
5. Submit the exact move payload
6. Reuse cached output if the same turn is repeated
7. Continue waiting for the next round or matchup

---

## Testing

Recommended baseline verification:

```bash
python -m py_compile agent.py
python -m pytest -q
python -m coverage run --branch -m pytest -q
python -m coverage report --include=agent.py -m
```

Required test areas:

- Configuration validation
- Payoff matrix
- History normalization
- Opponent isolation
- Match-boundary isolation
- Archetype probabilities
- PHANTOM handling
- Message credibility
- Finite-horizon rollout
- Strategic invariants
- Groq request formatting
- Structured output parsing
- Prompt-injection defence
- Unsafe-message replacement
- Groq failure to local fallback
- Deadline enforcement
- Cache idempotency
- Practice mode
- Test mode
- Official payload shape
- Submission timeout handling
- Secret redaction

Normal automated tests must not make live network calls.

### Optional live Groq test

A live smoke test should run only when explicitly enabled:

```env
RUN_LIVE_GROQ_TEST=1
```

It should verify authentication, model availability, structured output support, response parsing, and latency without printing secrets or unrestricted raw responses.

---

## Project Structure

The final runtime is intentionally simple:

```text
Technova_Agentic_Ai/
│
├── agent.py              # Complete production agent
├── README.md
├── .env.example
├── .gitignore
├── requirements.txt      # Only if a runtime dependency is truly needed
│
└── tests/                # Verification only; not required by the runtime
    ├── test_protocol.py
    ├── test_strategy.py
    ├── test_history.py
    ├── test_groq.py
    ├── test_safety.py
    ├── test_deadline.py
    └── test_simulations.py
```

The final `agent.py` must not import local project modules.

---

## How to Explain the Project to Judges

> Our system is a hybrid autonomous strategy agent for a seven-round Iterated Prisoner’s Dilemma. The arena sends the current round, opponent identity, PHANTOM state, and global history. The agent first normalizes and isolates the current opponent’s history. It then calculates behavioural statistics such as cooperation, retaliation, forgiveness, betrayal, and message credibility.
>
> Instead of assigning one fixed opponent label immediately, it maintains probabilities across eleven archetypes. A deterministic finite-horizon engine evaluates cooperation and defection across all remaining rounds and produces expected utilities, strategic evidence, and a recommended action.
>
> Groq is used only as a constrained final adjudicator when time permits. It sees the calculated candidates rather than raw history and must return strict structured JSON. Its answer is validated for schema correctness, fair play, and strategic consistency. If Groq fails, times out, or recommends a significantly inferior action, the local deterministic engine responds immediately.
>
> The system is protected against prompt injection, uses a bounded cache to prevent inconsistent resubmission, and reserves enough time to submit before the 25-second deadline. The final competition build is contained in one standalone `agent.py` so it can be deployed in a minimal judging environment.

### One-sentence explanation

> The agent learns the opponent’s strategy, calculates the expected value of both moves, uses Groq to adjudicate close decisions, validates the result, and always retains a fast deterministic fallback.

---

## Known Open Questions

The following must be confirmed with the organisers before the live event:

1. Final server URL
2. Assigned team ID and team token
3. Whether the disputed payoff is 0/5 or 0/6 from the opponent’s perspective
4. Exact `global_history` schema
5. Exact PHANTOM behaviour
6. Whether global history contains only our matches or tournament-wide records
7. Whether reasoning is visible only to judges or also to opponents
8. Whether final-round betrayal affects reputation in later matchups
9. Whether the full project directory is accepted or only one standalone `agent.py`

---

## Final Readiness Standard

The agent should be labelled **READY FOR COMPETITION** only when:

- The standalone file runs without local imports
- All tests pass
- Strategy and deadline branch coverage is at least 90%
- Real pipeline simulations meet documented targets
- Groq failure always falls back locally
- Every path returns a legal move and non-empty reasoning
- The complete decision leaves a safe submission reserve
- No secret is present in tracked files, Git history, logs, or test output
- Practice-server behaviour confirms the assumed protocol and history schema

Until then, the correct status is:

```text
READY FOR PRACTICE
```

---

## License and Competition Use

This repository is intended for the TECHNOVA 2026 Trust Arena competition. Ensure that all code, model usage, communication strategy, and deployment behaviour comply with the final organiser rules.
