# Protocol-sequence-v1 GPT-5.6 pilot — 2026-07-15

This research-only pilot replaced eight four-model-move continuation starts with
four eight-model-move starts while retaining the same planned maximum of 32 legal
model moves per configuration. Starts were frozen at game-like indices 0, 12, 24,
and 36—one from each category. The opponent remained the seeded random legal
engine. Every legal model and opponent move was scored at Stockfish depth 10, and
the exact production two-strike retry policy remained active.

The runner and selection were frozen before calls. Results are isolated in
`results/protocol_sequence.json` with `production_effect: none` in the manifest.
The machine-readable comparison is
`validation/2026-07-15-protocol-sequence-pilot.json`.

## Legality result

| Model | Core | Game-like | Prior 4-move continuation | New 8-move sequence | Live games |
| --- | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 1/50 (2.00%) | 1/48 (2.08%) | 1/32 (3.12%) | 3/32 (9.38%) | 24/381 (6.30%) |
| Terra low | 2/50 (4.00%) | 0/48 | 1/32 (3.12%) | 1/32 (3.12%) | 19/482 (3.94%) |
| Sol high | 1/50 (2.00%) | 0/48 | 1/32 (3.12%) | 0/30 | 3/311 (0.96%) |
| **Aggregate** | **4/150 (2.67%)** | **1/144 (0.69%)** | **3/96 (3.12%)** | **4/94 (4.26%)** | **46/1,174 (3.92%)** |

The new aggregate rate is close to the live aggregate and, unlike the prior
continuation sample, correctly orders the three discriminator models. Uncertainty
is still large: 95% Wilson intervals are 3.24–24.22% for Luna, 0.55–15.74% for
Terra, and 0–11.35% for Sol. This is strong pilot evidence, not a calibrated
production estimate.

The pilot made 94 first-attempt calls and three conditional retries, for 97 total
model calls. Sol's final sequence ended naturally after six model turns; the other
11 sequences reached their eighth turn or a production forfeit.

Luna's first illegal occurred on continuation turn 3 and recovered. Its later
illegals occurred on turns 7 and 8; the turn-8 event was the later second strike
that caused a forfeit. Terra's single turn-1 illegal recovered. Thus the longer
half directly contributed two of Luna's three events and the observed production
forfeit.

## CPL result

| Model | Turns 1–4 mean CPL | Turns 5–8 mean CPL | Overall raw mean | Overall 5000-cap mean | Median | 1000+ catastrophes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 74.6 | 1,890.9 | 953.4 | 580.0 | 74.0 | 3/31 (9.68%) |
| Terra low | 102.5 | 134.4 | 118.4 | 118.4 | 18.5 | 0/32 |
| Sol high | 26.0 | 80.6 | 51.5 | 51.5 | 1.5 | 0/30 |

Continuation CPL produced a much stronger discriminator than first-move quality:
the three models' first moves averaged 70.3 CPL for Luna, 38.8 for Terra, and 37.0
for Sol, while Luna's later half contained three mate-scale tactical collapses.
The raw Luna mean is consequently heavy-tailed; the 5000-CPL-capped mean preserves
the separation without allowing mate scores to dominate without bound.

Random-opponent mean CPL differed substantially—1,616 for Luna, 236 for Terra,
and 463 for Sol—because each model created a different sequence and random replies
sometimes entered mate-scale positions. This is genuine noise and interaction,
but it does not erase the evidence. Report raw, capped, median, catastrophe, and
turn-split measurements together rather than pretending one point mean is exact.

## Implications

1. Keep the longer four-by-eight structure as the leading continuation design.
   It added no planned calls and exposed both Luna's repeated illegality and its
   later chess-quality collapse.
2. Keep random legal replies and continuation CPL. Preserve opponent CPL and the
   pre-move evaluation so later models can account for stimulus severity.
3. Do not change production yet. Three configurations selected as discriminators
   are enough to validate the mechanism, not enough to fit a rating coefficient.
4. The next inexpensive analysis should test whether sequence CPL and legality
   improve held-out game-rating prediction using all evidence already collected.
   New cross-family calls remain the eventual independent validation.

## Candidate failure-position transfer experiment

The pilot produced four evolved failure events, not merely four starting
positions. Two arose from the sequence beginning at `game-like-025`, independently
tripping Luna medium and Terra low. These evolved FEN-plus-history states are a
promising candidate pool, together with live-game failure states.

Selection must measure transfer rather than memorization:

- preserve the complete legal move history and FEN at each failure;
- deduplicate equivalent states and exclude malformed source records;
- test each state on configurations other than the model that supplied it;
- compare against phase- and complexity-matched control states;
- rank candidates by leave-one-model/family-out illegal incidence;
- freeze the selected set before testing held-out families; and
- keep discovery, selection, and held-out results in separate artifacts.

This can deliberately enrich for positions that trip models while still giving
an honest estimate of whether failures transfer beyond their source model.

That experiment has now been completed. Foreign-source failure states were illegal
on 3/24 target attempts versus 0/24 matched controls. See
`validation/2026-07-15-failure-transfer-screen.md`; its three transfer-positive
candidates still require independent non-GPT-5.6 validation.
