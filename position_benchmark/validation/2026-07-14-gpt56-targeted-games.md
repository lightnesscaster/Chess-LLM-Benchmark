# GPT-5.6 targeted validation games — 2026-07-14

This follow-up added 16 color-balanced production games for the two GPT-5.6
configurations whose position predictions most strongly exceeded their initial
eight-game results. The games used the existing production player IDs and store.
Each target played three additional GPT-5.5 effort variants near its observed
rating and one additional game per color against the fixed 400-rated random bot.

## Added outcomes

| Target | Opponent | W-L-D | Target illegal forfeits |
| --- | --- | ---: | ---: |
| Luna medium | GPT-5.5 low | 0-1-1 | 1 |
| Luna medium | GPT-5.5 high | 0-2-0 | 2 |
| Luna medium | GPT-5.5 xhigh | 1-1-0 | 1 |
| Luna medium | random-bot | 2-0-0 | 0 |
| **Luna medium total** |  | **3-4-1** | **4** |
| Terra low | GPT-5.5 medium | 0-1-1 | 1 |
| Terra low | GPT-5.5 high | 2-0-0 | 0 |
| Terra low | GPT-5.5 xhigh | 1-1-0 | 1 |
| Terra low | random-bot | 2-0-0 | 0 |
| **Terra low total** |  | **5-2-1** | **2** |

Every target loss in the new batch was an illegal-move forfeit. Conditional on
not forfeiting, Luna scored 3-0-1 and Terra scored 5-0-1. This is not a separate
rating and should not be reported as one, but it establishes that the prediction
error is strongly tied to live-game legality rather than uniformly weak legal
chess moves.

## Legality evidence

| Target | New first-attempt illegals | New retry recovery | New later second strikes | All-game first-attempt illegal rate | All-game forfeits |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 10/174 (5.75%) | 6/6 | 4 | 24/381 (6.30%) | 11/16 |
| Terra low | 6/236 (2.54%) | 4/4 | 2 | 19/482 (3.94%) | 8/16 |

The retry prompt itself remained effective. All ten new eligible retries recovered;
the six new forfeits were later second strikes. Across all 112 GPT-5.6 games, 64
of 66 retries recovered, while 27 later second strikes caused forfeits. The
refreshed machine-readable audit is
`validation/2026-07-14-gpt56-game-retry-audit.json`.

Luna's live 6.30% first-attempt illegal rate remains far above its 2.00% core and
2.08% game-like rates. Terra's live 3.94% rate is close to its 4.00% core rate,
so static legality undermeasurement explains much more of Luna's error than
Terra's.

## Rating update

| Target | Production before | Production after | RD before | RD after | Validation RD-300 before | Validation RD-300 after | Current position prediction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 421 | 507 | 127 | 108 | 327 | 330 | 1,064 |
| Terra low | 578 | 593 | 137 | 103 | 297 | 475 | 980 |

The higher-RD validation target deliberately changes only the initial benchmark
seed uncertainty; production seeding remains unchanged. Its local artifact is
`validation/2026-07-14-gpt56-targeted-rd300.json`.

Terra's initial eight games materially overstated the prediction error: its
validation target rose 178 Elo. Luna's target was essentially unchanged. After
16 games each, the full position predictor is still optimistic by 734 Elo for
Luna and 505 Elo for Terra. The extra games reduced target uncertainty and did
not rescue the current prediction formula.

## Benchmark implications

1. Preserve first-attempt illegality and retry failure as separate quantities.
   Retry failure is rare; repeated first-attempt illegality is the dominant loss
   mechanism.
2. A static legality rate is not automatically transportable to live games.
   Luna demonstrates a large context shift even though core, game-like, and
   continuation rows all looked substantially safer.
3. Keep the frozen six-position legality-stress panel as an efficient diagnostic,
   but do not turn its noisy 1/6 samples directly into a rating coefficient.
4. The next protocol panel should use deterministic legal continuations and score
   legality over repeated game-like turns. This targets the observed context gap
   without using the random opponent's later-ply CPL as chess-quality evidence.
5. Do not change production coefficients from these two same-generation targets.
   Cross-family evidence is still required before fitting the mapping from static
   stress illegality to live-game survival.
