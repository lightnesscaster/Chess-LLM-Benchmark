# Frozen cross-family failure-transfer validation plan — 2026-07-17

This plan was frozen before any held-out target call. It tests the three
GPT-5.6-selected transfer-positive failure states against their three matched
controls. GPT-5.6 is excluded because it supplied both discovery and shortlist
selection evidence.

The panel remains research-only and has no production rating or acquisition
effect.

## Frozen workload

- Positions: `candidates/failure_transfer_positive_3.json` (SHA-256
  `5b35b5858b33428e2a4707f54460a2babc05c1e5ce736b1a1cdbdd54392901d1`)
- Three matched failure/control pairs per configuration
- Eight configurations from five provider families: Google, DeepSeek,
  Anthropic, Z.ai, and Qwen
- 48 first attempts, plus a production-protocol retry only after an illegal
  first answer
- Stockfish depth 30 for every CPL score
- OpenRouter backend
- Estimated cost: $0.1626 base, $0.3252 worst case; estimated-cost
  admission cap $0.33. This guard rejects an over-budget estimate before calls;
  it is not a runtime spend stop.

The exact model configurations and position pairs are recorded in
`2026-07-17-failure-transfer-heldout-plan.json`.

## Primary decision rule

The primary endpoint is paired first-answer illegality over 24 matched pairs.
The directional hypothesis is that failure states have higher illegality than
their controls. The test is a one-sided exact McNemar test over discordant pairs.

The shortlist passes only if all of the following hold:

1. The aggregate failure illegal rate is greater than the control rate.
2. The one-sided exact McNemar probability is at most 0.05.
3. Failure-state illegals outnumber control illegals within at least three
   distinct provider families.

Passing promotes the shortlist only to a broader calibration candidate. It does
not change the production rating formula, automatic acquisition policy, or any
published benchmark score. Failure keeps the shortlist research-only and does
not justify expanding failure-position mining.

## Secondary evidence

Retry recovery, family- and target-level illegality, candidate-level transfer,
and depth-30 CPL (all rows and legal-only) are descriptive secondary endpoints.
They cannot reverse the primary decision.

This is a replication test of a deliberately selected six-position shortlist.
It is not an unbiased estimate of how often arbitrary live-game failures transfer
to other model families.
