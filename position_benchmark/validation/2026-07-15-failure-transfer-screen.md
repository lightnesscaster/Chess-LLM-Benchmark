# Cross-model failure-position transfer screen — 2026-07-15

This research screen tested whether positions where one GPT-5.6 base-model line
made a well-formed illegal chess move would trip different GPT-5.6 base-model
lines more often than matched legal controls.

The matrix was frozen before target calls. Twelve live-game failure states were
sampled: four each from Luna, Terra, and Sol, spanning movement/blocked-path,
empty-source, king-safety, and own-piece failures where available. Each was paired
with a legal state from the same source configuration and a different game,
matched on side, phase, history length, castling rights, capture context, material,
piece count, and legal-move count.

Each representative target saw only source states from the other two entire base
model lines:

- Luna medium saw Terra- and Sol-sourced states;
- Terra low saw Luna- and Sol-sourced states; and
- Sol high saw Luna- and Terra-sourced states.

Thus no target was tested on a failure supplied by another reasoning level of its
own base model. The frozen matrix is
`candidates/failure_transfer_screen_v1/matrix.json`.

## Depth correction

An initial command mistakenly used Stockfish depth 10. It was interrupted after
only two Luna rows completed; Terra and Sol retained no completed rows. Those
partial `/tmp` artifacts were discarded and are absent from all results and
analysis. The same pre-frozen candidate IDs were rebuilt at depth 30, and every
official row was then rerun from scratch and stamped `stockfish_depth: 30`.

## Transfer result

| Target | Foreign-source failures | Matched controls |
| --- | ---: | ---: |
| Luna medium | 2/8 illegal | 0/8 illegal |
| Terra low | 1/8 illegal | 0/8 illegal |
| Sol high | 0/8 illegal | 0/8 illegal |
| **Aggregate** | **3/24 = 12.5%** | **0/24 = 0%** |

The failure-state Wilson interval is 4.34–31.00%; the control interval is
0–13.80%. All three discordant matched pairs favored the discovered failure state
and none favored the control, but the one-sided exact McNemar probability is still
0.125. The direction and magnitude are promising; 24 pairs are not enough to
claim a stable population effect.

All three illegal first answers recovered under the production retry prompt.

## Which failures transferred

| Candidate | Source | Original failure type | Foreign targets illegal |
| --- | --- | --- | ---: |
| `failure-transfer-luna-003` | Luna medium | King safety | 1/2 |
| `failure-transfer-sol-001` | Sol low | Blocked/movement rule | 1/2 |
| `failure-transfer-sol-003` | Sol medium | King safety | 1/2 |

The other nine candidates were legal on both foreign targets. Luna-sourced states
transferred on 1/8 target attempts, Terra-sourced states on 0/8, and Sol-sourced
states on 2/8. These are not deterministic traps: every positive candidate failed
on one foreign target and passed on the other.

## Depth-30 CPL evidence

| State type | Mean CPL | 5000-cap mean | Median | Legal-only mean |
| --- | ---: | ---: | ---: | ---: |
| Discovered failures | 650.8 | 631.4 | 28.0 | 68.6 |
| Matched controls | 27.5 | 27.5 | 2.5 | 27.5 |

The all-row mean includes the benchmark's illegal-move CPL penalty. More
importantly, legal answers on discovered failure states still averaged 68.6 CPL
versus 27.5 on controls. By target, legal-only failure/control means were 28.3/24.0
for Luna, 127.7/40.0 for Terra, and 47.1/18.6 for Sol. The mined states therefore
showed some chess-quality difficulty beyond the three illegal answers.

## Decision and next gate

Failure-state mining works well enough to continue: it raised held-out-within-line
illegal incidence above both the matched controls and the earlier independent
game-like panel. But selecting the three transfer-positive candidates used these
GPT-5.6 target results, so GPT-5.6 cannot validate the resulting shortlist again.

`candidates/failure_transfer_positive_3.json` freezes the three positives and
their three matched controls. It is explicitly marked `production_effect: none`
and `selection_uses_gpt56_transfer_results: true`. The next valid test must use
non-GPT-5.6 model families that supplied neither discovery nor selection evidence.
Only after that held-out test should we decide whether to expand failure mining,
promote a compact legality panel, or use transferred difficulty as an input to the
rating predictor.
