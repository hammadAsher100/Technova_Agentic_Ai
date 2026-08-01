# Trust Arena Architecture — Judge Brief

The official protocol is simultaneous: both agents submit a move and message blind, and both are revealed together. `opponent_message` is therefore `None` during the current decision. Messages can influence later rounds, rematches, and reputation, never the opponent's already-locked current move.

## Cognitive pipeline

```text
game_state -> validation/normalization -> opponent profile -> deterministic patterns
-> normalized archetype beliefs -> finite-horizon candidate utilities -> compact prompt
-> Groq structured decision -> strategic/fair-play critic -> Gemini on failure
-> deterministic emergency policy -> exact official tuple
```

`global_history` is authoritative. The tolerant normalizer accepts documented fields and evidence-backed aliases, handles either fixed-perspective or explicit two-participant records, preserves missing actions as missing, ignores unknown fields, and emits warning counts without crashing. It rebuilds the current profile each turn. No JSON reputation file is required.

The opponent model scores Pacifist, Predator, mirror/tit-for-tat, generous tit-for-tat, grim trigger, win-stay/lose-shift, alternator, random, opportunist, endgame betrayer, and strategic unknown. Action evidence dominates message evidence. Message credibility is learned only by comparing a statement with the subsequent move. PHANTOM mode flattens identity beliefs, lowers confidence, and favors conservative behavior-based adaptation.

The planner computes immediate payoffs and remaining-round comparable utility for both moves, including retaliation, repair, exploitation, and reputation. The LLM sees those calculations, top beliefs, and at most three normalized/sanitized messages—not raw history—and selects between candidates. Groq `openai/gpt-oss-120b` is always primary; Gemini `gemini-3.6-flash` is called only after timeout, rate limit, provider failure, or invalid structure. Network timeouts use remaining monotonic budget. A local policy returns in milliseconds when fallback time is insufficient.

Strict parsing rejects prose, Markdown fences, unknown fields, invalid moves, empty/oversized reasoning, and invalid confidence. Outbound messages are scanned for override commands, role reassignment, fake system framing, prompt extraction, jailbreak language, and secret references. Unsafe messages are replaced without throwing away a valid move. Repeated delivery of one turn uses a fingerprint of match, round, opponent, and history and resubmits an identical cached payload.

## Operations and open question

Start with `python agent.py`. The root executable preserves `X-Team-Token`, `/my-turn`, `/my-move`, their `/practice` counterparts, `wait`, `your_turn`, `match_complete`, `practice_mode`, and organizer-controlled `test_mode`.

The rulebook says cooperation versus opponent defection scores `(0, 6)` while the scaffold says `(0, 5)`. The centralized default currently follows the rulebook. Organizers must resolve this, along with the confirmed server URL and assigned team credentials, before deployment.
