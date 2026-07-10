# Position Benchmark: Canonical Production Specification

This document is the source of truth for the position benchmark used to predict a
model's game-playing rating. The production design was finalized on June 23, 2026
and implemented in commit `08ffc44` (`improve position benchmark prediction`).

The benchmark has one required core and optional, downside-only diagnostics. A
supplemental panel is not part of the core workload and can never raise a rating.
Experimental position sets and analysis scripts do not change this specification
unless this document, the benchmark version, and the validation record are updated
together.

## Production decision

The required benchmark is exactly **50 equal positions** from
`panels/core_equal_50.json`.
Each position must:

- be presented with its recorded move history replayed onto the board;
- use the normal chess prompt and the same model/provider surface being rated;
- be scored against Stockfish at depth 30;
- record exact-best, centipawn loss (CPL), and first-attempt legality; and
- be stamped `position_benchmark_version: history-replay-v2`,
  `prompt_history_replay: true`, and `stockfish_depth: 30` or greater.

The history replay is essential. A board reconstructed from FEN alone has an empty
move stack and does not reproduce the context a model receives during a game.
Legacy rows without the freshness markers remain useful for research, but they are
not production-ready evidence.

The core workload is **50 model calls per model configuration**. For 12 model
configurations, such as three GPT model variants at four reasoning levels, the core
is 600 calls. The optional panels below are additional work and must not be included
in that count unless they are explicitly requested.

## Artifact layout

`benchmark_manifest.json` is the machine-readable directory of active artifacts:

| Purpose | Positions | Results | Status |
| --- | --- | --- | --- |
| Rating-prediction core | `panels/core_equal_50.json` | `results/core.json` | Required |
| Non-opening game-like | `panels/game_like_48.json` | `results/game_like.json` | Optional downside only |
| Continuation stability | First 8+ game-like positions | `results/stability.json` | Optional downside only |
| Historical blunders | `panels/optional_blunder_25.json` | `results/optional_blunder.json` | Optional historical |

Every selected position has a stable `position_id` and a panel-local
`position_idx`. Core IDs are `core-equal-001` through `core-equal-050`. Numeric
indices from the former combined file are retained only as `legacy_position_idx`.
Candidate pools live under `candidates/`; old mixed files live under `legacy/` and
are never active inputs.

The runner refuses any positions file marked `active_production_input: false`.
Historical tools may override that protection only by explicitly passing
`--allow-legacy-input`; normal benchmark commands must never use that flag.

## Rating formula

For the 50 equal positions, calculate:

- `mean CPL`: arithmetic mean CPL across the 50 positions;
- `best %`: percentage of responses exactly matching Stockfish's best move; and
- `legal %`: percentage legal on the first attempt.

Let `p = 1 - legal_pct / 100`. The probability of surviving 40 model turns with
at most one illegal response is:

```text
S40 = 100 * ((1 - p)^40 + 40 * p * (1 - p)^39)
```

The base rating prediction is:

```text
predicted_rating =
    1298.57
    - 200.43 * ln(mean_CPL + 1)
    + 15.39 * best_pct
    + 5.85 * S40
```

The implementation is `predict_rating()` in `predictions.py`. Prediction error in
the validation reports is always defined as:

```text
error = actual_game_rating - predicted_rating
```

A negative error therefore means the position benchmark overestimated the model.

## Optional downside checks

These panels address specific optimistic failure modes. They are not required for a
production-ready core result, do not provide positive evidence, and never increase
the core estimate.

### Non-opening game-like panel

The game-like panel contains 48 non-opening positions: 12 quiet/equal, 12 tactical/
equal, 12 advantage-conversion, and 12 defensive positions. It uses the same rating
formula, history replay, and depth-30 freshness requirements. CPL is capped at 5000
for this supplemental estimate.

The game-like prediction replaces the current estimate only when it is **more than
150 Elo lower**:

```text
if game_like_prediction + 150 < current_prediction:
    current_prediction = game_like_prediction
```

Its workload is 48 additional model calls per configuration.

### Continuation-stability probe

The stability probe starts from representative non-opening positions and lets the
model play one side for eight plies against the seeded random engine. It exercises
the real illegal-move retry and forfeit path. The finalized validation used
Stockfish depth 10 to score the model's moves.

A probe is eligible when it has at least eight attempted starting positions and at
least 24 scored/model-attempted moves, unless repeated forfeits already provide
sufficient evidence. Define:

- `F`: percentage of starting positions ending in a model illegal-move forfeit;
- `K`: percentage of scored model moves losing at least 1000 CPL.

If `F + K < 5`, the stability probe does nothing. Otherwise its downside estimate
is:

```text
stability_cap = clamp(650 - 6*F - 8*K, -500, 650)
```

As with the game-like panel, the cap is applied only when it is more than 150 Elo
below the current estimate. Eight starting positions require up to 32 ordinary
model turns, plus any illegal-move retries.

### Historical blunder panel

`panels/optional_blunder_25.json` contains 25 curated blunder positions. The
predictor can consume fresh results as another one-way downside check using the same
150-Elo trigger and 5000-CPL cap. It is **not part of the required core**. June
analysis found that blunder positions added almost no useful predictive value, so
they should not be run by default or described as part of the production call count.

Their presence is retained for backward compatibility and research. Running them
adds 25 model calls per configuration.

During the stable-ID migration, 361 old optional-blunder result rows were excluded
because their stored FEN did not match their claimed numeric index. Thirteen model
sets retained all 25 valid rows; nineteen retained only six. Incomplete sets cannot
pass the 25-position readiness check and therefore cannot affect production ratings.

The predictor evaluates available caps in this order: historical blunder panel,
game-like panel, then continuation stability. Each comparison uses the estimate
remaining after the preceding check.

## June 2026 validation record

The end-of-session validation recorded the following performance after adding the
history-replayed core and the downside-only checks:

| Cohort | Rows | MAE | RMSE | High-rated failures |
| --- | ---: | ---: | ---: | ---: |
| Fully refreshed validation rows | 12 | 148.0 | 217.1 | 0 of 6 outside +/-200 Elo |
| Reliable historical/readiness audit | 43 | 164.5 | 235.1 | 0 of 6 outside +/-200 Elo |

The stability cap corrected three known optimistic estimates:

| Model | Before cap | After cap | Actual at validation |
| --- | ---: | ---: | ---: |
| Gemini 2.5 Flash (no thinking) | 803 | 547 | 368 |
| Gemini 3.1 Flash Lite Preview | 802 | 408 | 278 |
| GPT-5.1 (no thinking) | 747 | 495 | 508 |

These are a frozen validation snapshot, not live leaderboard metrics. Ratings can
move as games are added or recalculated. The reproducible raw-data snapshot is git
commit `08ffc44`; current files may contain later games and models. A machine-readable
copy of the frozen measurements is stored in
[`validation/2026-06-23.json`](validation/2026-06-23.json).

## GPT-5.5 result and limitation

The June snapshot is easy to misremember because one GPT-5.5 row was extraordinarily
accurate while the family as a whole was not. Predictions below are recomputed from
the raw position rows in commit `08ffc44`; actual ratings come from that commit's
`data/ratings.json`.

| Configuration | Predicted | Actual | Error (actual - predicted) | Fresh core row? |
| --- | ---: | ---: | ---: | --- |
| GPT-5.5 no thinking | 486.692 | 185.760 | -300.932 | No |
| GPT-5.5 low | 1132.289 | 555.009 | -577.279 | **Yes** |
| GPT-5.5 medium | 1033.772 | 411.626 | -622.146 | No |
| GPT-5.5 high | 437.812 | 437.455 | **-0.356** | No |
| GPT-5.5 xhigh | 1484.999 | 634.693 | -850.306 | No |

The nearly zero error belongs specifically to the legacy GPT-5.5 high row. It must
not be presented as proof that the finalized benchmark predicted all GPT-5.5
reasoning levels accurately. The fresh GPT-5.5 low result remained a major miss.
A separate OpenRouter stability probe for GPT-5.5 low was stable, so it was not
appropriate to force the rating down with the continuation cap; provider/surface
differences or game-pool effects remained plausible explanations.

## Running and auditing

Run the required core for one configured model with the provider surface that will
be used in games:

```bash
python position_benchmark/run_benchmark.py \
  --players "PLAYER_ID" \
  --depth 30 \
  --api codex
```

Use `--limit 1` and a temporary `--output` for a smoke test. Do not merge smoke-test
rows into the production results file.

Run the optional game-like panel only when that diagnostic is wanted:

```bash
python position_benchmark/run_benchmark.py \
  --positions position_benchmark/panels/game_like_48.json \
  --output position_benchmark/results/game_like.json \
  --players "PLAYER_ID" \
  --depth 30 \
  --api codex
```

Run the optional finalized continuation probe with:

```bash
python scripts/run_stability_probe.py \
  --players "PLAYER_ID" \
  --limit 8 \
  --probe-plies 8 \
  --score-depth 10 \
  --api codex
```

Audit freshness, coverage, prediction error, and any available supplemental caps:

```bash
python scripts/audit_position_benchmark_readiness.py --show-all
```

The audit reads the separate optional result files under `position_benchmark/results/`
when they exist. A current leaderboard audit will not necessarily reproduce the
frozen June metrics because actual game ratings continue to change.

## Change control

Any future change to the production methodology must include all of the following:

1. Change `CURRENT_BENCHMARK_VERSION` when prompts, position selection, scoring, or
   required inputs change in a result-invalidating way.
2. Update this document and the constants/tests in the same commit.
3. Record the exact validation cohort, commit, MAE, RMSE, error definition, and
   target-failure count.
4. Separate current rows from legacy or provider-mismatched rows.
5. State the required and optional call counts explicitly.
6. Preserve the model API/provider surface between the position benchmark, optional
   probes, and rated games, or label the comparison as provider-mismatched.

Research scripts and newly generated panels are experiments until this change
control is completed.
