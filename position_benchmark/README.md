# Position Benchmark: Canonical Production Specification

This document is the source of truth for the position benchmark used to predict a
model's game-playing rating. The production design was finalized on June 23, 2026
and implemented in commit `08ffc44` (`improve position benchmark prediction`).

The benchmark has one required core and automatic, downside-only supplements. A
supplemental panel is not required for core rating readiness and can never raise a rating.
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
- record exact-best, centipawn loss (CPL), and first-attempt legality;
- after an illegal first answer, issue the production illegal-move retry and
  record whether the second answer is legal; and
- be stamped `position_benchmark_version: history-replay-v2`,
  `prompt_history_replay: true`, and `stockfish_depth: 30` or greater.

The history replay is essential. A board reconstructed from FEN alone has an empty
move stack and does not reproduce the context a model receives during a game.
Legacy rows without the freshness markers remain useful for research, but they are
not production-ready evidence.

The core workload has **50 required first-attempt calls per model configuration**,
plus exactly one conditional retry call for each illegal first answer. It therefore
uses 50–100 total calls per configuration. For 12 configurations, the base is 600
calls and the exact total is `600 + the number of first-attempt illegals` (at most
1200). That remains the correct count when discussing the core alone.

The scheduler's complete automatic acquisition suite is larger: 50 core calls, 48
game-like calls, and up to 32 first-attempt continuation turns, or **130 planned
first-attempt calls per configuration**. Conditional retries add 0–50 core calls,
0–48 game-like calls, and at most eight continuation calls. The automatic suite is
therefore 130–236 calls per configuration; for 12 configurations its planned base
is 1,560 calls, not 600.

The retry call uses the same player method and arguments as a production game:
the unchanged history-replayed board, `is_retry=True`, and
`last_move_illegal=<first response or "invalid">`. Rows are stamped
`conditional_retry_protocol_version: production-game-retry-v1`. Canonical CPL,
best-move, and legality fields continue to describe the first answer; retry fields
are additional evidence and do not currently change the production rating formula.
Summary `conditional_retry` metrics are calculated only over stamped rows, so old
results remain readable without being mistaken for measured recovery evidence.
Per-row `prompt_tokens` and `completion_tokens` include both chess prompts, while
the `initial_*` and `retry_*` fields retain the split. CLI cost estimates are
empirical: same-player history is preferred, followed by same-model-and-effort,
effort-level P90, and reasoning/non-reasoning P90 evidence. The CLI prints the
assumed prompt/completion tokens per call, source, and sample count. With no saved
history, the conservative fallback is 15,000 completion tokens for reasoning and
500 for non-reasoning calls; the former 200-token reasoning default was removed.
The static runner labels the first-attempt estimate as the base and also reports
the all-conditional-retries upper bound. OpenRouter usage from truncated responses
is retained before provider-level retry when the response supplies usage, so
recorded actual cost includes those otherwise discarded generations.

## Artifact layout

`benchmark_manifest.json` is the machine-readable directory of active artifacts:

| Purpose | Positions | Results | Status |
| --- | --- | --- | --- |
| Rating-prediction core | `panels/core_equal_50.json` | `results/core.json` | Required |
| Non-opening game-like | `panels/game_like_48.json` | `results/game_like.json` | Automatic downside only |
| Continuation stability | 8 stratified game-like positions | `results/stability.json` | Automatic downside only |
| Long-sequence continuation | 4 stratified game-like positions | `results/protocol_sequence.json` | Research candidate |
| Historical blunders | `panels/optional_blunder_25.json` | `results/optional_blunder.json` | Optional historical |

Every selected position has a stable `position_id` and a panel-local
`position_idx`. Core IDs are `core-equal-001` through `core-equal-050`. Numeric
indices from the former combined file are retained only as `legacy_position_idx`.
Candidate pools live under `candidates/`; old mixed files live under `legacy/` and
are never active inputs.

The runner refuses any positions file marked `active_production_input: false`.
Historical tools may override that protection only by explicitly passing
`--allow-legacy-input`; normal benchmark commands must never use that flag.

## Automatic acquisition policy

`benchmark_manifest.json` defines `production-supplements-depth30-v2`. For every model
selected in a normal benchmark run that is configured, non-engine, and not frozen,
the scheduler checks current readiness and acquires missing panels in this fixed
order:

1. `core`;
2. `game_like`;
3. `continuation_stability`.

This is not an operator choice. A model with a valid core but missing or stale
supplements is still queued. Static rows must have history replay, the required
Stockfish depth, and complete position coverage. Every new acquisition records the
production retry protocol; otherwise-current historical static rows are not
invalidated solely because retry recovery was not measured, since it remains
additive evidence and does not change their rating formula. Continuation results
must use the exact stratified indices, Stockfish depth 30, zero API-error games,
and the current retry stamp. Depth-10 continuation rows are historical research
evidence only and cannot affect a production prediction.

The budget is checked before each panel. If the next panel does not fit, the
scheduler preserves completed work and resumes at the first missing panel on a
later run. A model that acquires any panel is deferred from ordinary games for that
run, preserving the existing one-benchmark-contribution scheduling behavior.
Historical blunders and every research-candidate panel are explicitly excluded
from automatic acquisition.

When all three automatic panels first become ready, the scheduler also locks the
fixed continuation-cap candidate predictions in
`results/stability_cap_shadow.json` before scheduling that configuration's first
game. This shadow record has `production_effect: none`; it cannot seed or modify a
rating. Configurations already used in the frozen development audit are excluded
from prospective enrollment.

## Rating formula

For the 50 equal positions, calculate:

- `mean CPL`: arithmetic mean CPL across the 50 positions;
- `best %`: percentage of responses exactly matching Stockfish's best move; and
- `legal %`: percentage legal on the first attempt.

Let `p = 1 - legal_pct / 100`. The probability of surviving 40 model turns with
at most one first-attempt illegal is:

```text
S40 = 100 * ((1 - p)^40 + 40 * p * (1 - p)^39)
```

This current production proxy implicitly assumes the retry always succeeds. Under
the actual two-strikes-per-game runner, if `q` is the probability that the one
eligible retry also fails, exact survival is:

```text
S40(p, q) = 100 * ((1 - p)^40 + 40*p*(1 - p)^39*(1 - q))
```

The per-turn expression `(1 - p*q)^40` would apply only if every later illegal
first answer received a fresh retry. Production does not do that: after a recovered
first strike, a later second illegal response forfeits immediately. The validated
rating coefficients still use the original `q=0` proxy; the research metric must
not replace it until retry evidence and coefficients are validated together.

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

## Automatic downside checks

These panels address specific optimistic failure modes. The scheduler acquires them
automatically, but they are not required for a production-ready core result, do not
provide positive evidence, and never increase the core estimate. This distinction
allows historical core-only results to remain readable without making acquisition
for new or actively selected models manual.

### Non-opening game-like panel

The game-like panel contains 48 non-opening positions: 12 quiet/equal, 12 tactical/
equal, 12 advantage-conversion, and 12 defensive positions. It uses the same rating
formula, history replay, and depth-30 freshness requirements. CPL is capped at 5000
for this supplemental estimate.

Its legality input is deliberately panel-aware. Calculate first-attempt legality
for the fresh core and game-like panels, plus move-attempt legality from a fresh
continuation probe when available. The pooled rate is reported as a diagnostic,
but the game-like prediction uses the lowest eligible panel rate. A simple pooled
average is not used for scoring because strong legality in one context could hide
a severe failure in another. Supplemental legality can therefore lower the
game-like estimate but can never make it more optimistic.

The game-like prediction replaces the current estimate only when it is **more than
150 Elo lower**:

```text
if game_like_prediction + 150 < current_prediction:
    current_prediction = game_like_prediction
```

Its workload is 48 additional first-attempt calls per configuration, plus one retry
for every illegal first answer (48–96 total calls).

### Continuation-stability probe

The stability probe starts from eight representative non-opening positions and
lets the model play one side for eight plies against the seeded random engine. The
default selector deterministically takes two positions from each game-like bucket
in round-robin order: advantage conversion, defense, quiet/equal, and tactical/
equal. With the current ordered panel, those indices are `0, 12, 24, 36, 1, 13,
25, 37`. It exercises the real illegal-move retry and forfeit path. Stockfish depth
30 scores every continuation move, matching the core and game-like panels.

Eligible summaries must be stamped
`stability_probe_version: stratified-depth30-v3`.
Earlier summaries selected positions 0 through 7, all from the advantage-conversion
block, and are retained only as historical research evidence. They cannot affect a
current production prediction; affected models need an eight-start rerun under the
stratified policy. Existing canonical PGNs can instead be rescored without model
calls:

```bash
python scripts/run_stability_probe.py \
  --rescore-existing \
  --score-depth 30 \
  --workers 4
```

A probe is eligible when it has at least eight attempted starting positions and at
least 24 scored/model-attempted moves, unless repeated forfeits already provide
sufficient evidence. Define:

- `F`: percentage of starting positions ending in a model illegal-move forfeit;
- `K`: deduplicated catastrophe events divided by scored model moves, where each
  continuation start contributes at most one event after its first move losing at
  least 1000 CPL. Later catastrophe-scale moves in the same trajectory are treated
  as correlated consequences, not independent failures.

If `F + K < 5`, the stability probe does nothing. Otherwise its downside estimate
is:

```text
stability_cap = clamp(650 - 6*F - 8*K, -500, 650)
```

This trajectory deduplication is a conservative counting correction: it preserves
the original exposure denominator and coefficients and can never make the cap more
severe. A July 21 depth-30 audit compared the old move count, this correction,
survival-style hazard censoring, repeated-forfeit-only protection, and no hard cap
against both RD-300 and no-position-seed game ratings. The final development
cohort contains 30 configurations, 17 model-line families, and eight labs, so its
30/eight acquisition gate passes. However, the leading
`repeated_forfeit_only` candidate changes only three configurations, all from
OpenAI. Its family- and lab-bootstrap intervals cross zero, and excluding the
influential GPT-3.5 Turbo Instruct family makes it slightly worse than the actual
production reference, `deduplicated_move_exposure_cap`, on both targets.
Therefore no coefficient refit, hazard redesign, or cap removal was adopted. See
`validation/2026-07-21-depth30-stability-cap-analysis.md` and the immutable
development-cohort declaration in
`validation/2026-07-23-depth30-cap-development-freeze.json`.

As with the game-like panel, the cap is applied only when it is more than 150 Elo
below the current estimate. Eight starting positions require up to 32 ordinary
model turns, plus any illegal-move retries.

### Prospective continuation-cap holdout

Future configurations are evaluated under the fixed
`validation/depth30-cap-prospective-policy.json` contract. A configuration is
eligible only when it was not in the development cohort and its candidate
predictions were locked after all three current automatic panels became ready and
before its first game. The append-only shadow ledger preserves the predictions and
input fingerprints; later game outcomes never rewrite them.

This timing is enforced, not advisory. Both normal scheduling and saved manual
games fail closed for a zero-game configuration if the shadow record cannot be
written. Configurations with pre-existing games remain legacy/non-prospective.
An intentional saved-manual-game exception requires
`--allow-nonprospective-save`, which writes a persistent exclusion before the
game starts; `--no-save` creates no rating evidence and needs no exception.

The primary outcome is an isolated no-position-seed game rating with at least
eight games and games RD no greater than 200. The comparison includes only mature
configurations whose challenger prediction differs from production. Evaluation
does not open until it has at least 12 mature holdouts, eight affected holdouts,
six affected families, and four affected labs. Promotion then requires every
predeclared MAE, RMSE, bias, family-bootstrap, lab-bootstrap, and
leave-one-lab-out check to pass. Validation recalculation sorts source games and
players deterministically, uses a local seeded shuffle, normalizes timestamps to
the latest included game, and serializes rating keys in sorted order.

Affected prospective configurations receive a fixed 2x scheduling-priority
multiplier until they reach the eight-game/RD-200 maturity gate. The priority uses
only locked prediction disagreement, game count, and games RD—never wins, losses,
scores, or rating movement. Inspect current progress without changing production:

```bash
python cli.py cap-holdout-status
```

Refresh the isolated target and evaluate the locked ledger with:

```bash
python cli.py recalculate \
  -c config/benchmark.yaml \
  --validation-output position_benchmark/validation/depth30-cap-prospective-no-position-ratings.json \
  --validation-seed-rd 350 \
  --validation-disable-benchmark-seeds
```

That exact isolated recalculation automatically runs the evaluator. It can also be
rerun without recalculating ratings via
`python scripts/evaluate_stability_cap_holdout.py`. The evaluator writes the
current machine-readable decision and report under
`position_benchmark/validation/`. Every status remains research-only unless the
fixed promotion gate passes and a separate production change is explicitly made.

### Family-held-out supplemental predictor audit

The 2026-07-17 zero-call audit replayed 6,276 production games using the current
benchmark prior mean with validation-only RD 300, then evaluated learned candidates
with nested leave-one-model-line-family-out prediction. Production remained at RD
166 and was not written.

On the broader game-like cohort (15 configurations, six families), the current
downside system materially improved over the core alone: MAE fell from 354.8 to
239.1 Elo and RMSE from 476.7 to 323.0. No learned aggregate, category, legality,
or continuous-deadband replacement improved held-out MAE. Pooled first-answer
illegality was the strongest remaining cross-family diagnostic (Spearman -0.617
with `actual - predicted`), but fitting it still worsened MAE.

The July 17 audit used twelve GPT-5.6 `stratified-v2` continuation records scored
at depth 10. Those rows are now historical until depth-30 rescoring. Within that
selected cohort,
continuation CPL had almost no relationship to remaining prediction error
(Spearman -0.091) and did not improve held-out MAE. The earlier GPT-5.6 defense-CPL
signal also failed to generalize: its correlation was -0.643 inside GPT-5.6 but
only -0.157 across the broader game-like cohort. The direct continuation
forfeit/catastrophe cap made no additional adjustment for these twelve models;
continuation legality still participates in the game-like panel's conservative
legality input.

The frozen evidence gate required 30 configurations and eight families, so the
decision is to retain the production predictor. See
`validation/2026-07-17-supplemental-predictor-analysis.md` and its frozen plan for
the full leakage controls and results.

Continuation rows explicitly record the outcome of the first production retry in
each probe game as recovery, failure, or unknown provider outcome. A later second
illegal strike is an immediate forfeit under the game policy and is not counted as
another retry opportunity. Saved rows can be deterministically backfilled without
model calls because their illegal-event ply numbers and PGNs establish whether the
retry produced a legal move:

```bash
python scripts/run_stability_probe.py --backfill-retry-metrics
```

### Experimental long-sequence continuation

`protocol-sequence-v1` keeps the continuation workload near the existing 32-turn
budget but uses four starts with up to eight model turns each. It selects indices
`0, 12, 24, 36`, retains seeded random legal replies, scores both sides at
Stockfish depth 30, and records exact first-attempt illegality separately from
conditional retry responses. Results live in `results/protocol_sequence.json` and
have no production effect.

The initial Luna-medium, Terra-low, and Sol-high pilot produced 4/94 first-attempt
illegals versus 46/1,174 in their live games, while also separating their later
CPL sharply. This is promising mechanism evidence from three deliberately chosen
configurations, not enough to fit or activate a new rating rule. See
`validation/2026-07-15-protocol-sequence-pilot.md`.

Run the frozen research protocol with:

```bash
python scripts/run_stability_probe.py \
  --protocol-sequence-v1 \
  --players "PLAYER_ID" \
  --api codex \
  --allow-unknown-cost
```

### Historical blunder panel

`panels/optional_blunder_25.json` contains 25 curated blunder positions. The
predictor can consume fresh results as another one-way downside check using the same
150-Elo trigger and 5000-CPL cap. It is **not part of the required core**. June
analysis found that blunder positions added almost no useful predictive value, so
they should not be run by default or described as part of the production call count.

Their presence is retained for backward compatibility and research. Running them
adds 25 first-attempt calls per configuration; the shared runner also records a
conditional retry after any illegal first answer.

During the stable-ID migration, 361 old optional-blunder result rows were excluded
because their stored FEN did not match their claimed numeric index. Thirteen model
sets retained all 25 valid rows; nineteen retained only six. Incomplete sets cannot
pass the 25-position readiness check and therefore cannot affect production ratings.

The predictor evaluates available caps in this order: historical blunder panel,
game-like panel, then continuation stability. Each comparison uses the estimate
remaining after the preceding check.

### Experimental legality-stress candidate

`candidates/legality_stress_6.json` is a six-position subset of the core selected
using 14 pre-GPT-5.6 configurations across nine model families. Selection averaged
illegal rates equally across families and used a predeclared 25% threshold; GPT-5.6
was held out completely. On the 12 held-out GPT-5.6 configurations it raised
first-attempt illegal incidence from 3.50% on the full core to 9.72%, a 2.78x lift,
while its model-level illegal rate correlated 0.730 with live-game first-attempt
illegality versus 0.628 for the full core.

This is an efficient research panel for collecting retry and high-risk legality
evidence, not a replacement core and not production rating evidence. It requires
six base calls plus failure-dependent retries per configuration. Its frozen
selection metadata and held-out diagnostics are embedded in the candidate JSON.

The initial three-configuration discriminator batch was run with:

```bash
python position_benchmark/run_benchmark.py \
  --positions position_benchmark/candidates/legality_stress_6.json \
  --output position_benchmark/results/legality_stress.json \
  --players "gpt-5.6-luna (medium)" "gpt-5.6-terra (low)" "gpt-5.6-sol (high)" \
  --depth 30 \
  --api codex
```

This is 18 first-attempt calls plus conditional retries. The output is deliberately
separate from production core results and has no production rating effect.
It produced two initial illegals, both recovered on retry. The same three models
had also produced two illegals on these six positions in their original core runs,
although neither failure repeated at the same model-position pair. See
`validation/2026-07-14-legality-stress-pilot.md` for the repeatability analysis and
the limits on interpreting this small stress sample.

### Experimental failure-transfer shortlist

`candidates/failure_transfer_positive_3.json` contains three live-game failure
states selected for transfer within GPT-5.6 and three matched controls. Because
GPT-5.6 supplied discovery and selection evidence, its required replication used
eight configurations from five non-GPT-5.6 families. The plan, targets, primary
test, and promotion threshold were frozen before the 48 first attempts.

The held-out panel produced 6/24 illegal first answers on failure states and 4/24
on controls. Four discordant pairs favored the failure states and two favored the
controls, giving a one-sided exact McNemar probability of 0.34375. Only two of the
five families had a positive direction, below the frozen requirement of three.
The shortlist therefore failed its promotion gate and remains research-only. It
must not affect production ratings or automatic acquisition. See
`validation/2026-07-17-failure-transfer-heldout.md` for the full result and
`validation/2026-07-17-failure-transfer-heldout-plan.json` for the frozen design.

## June 2026 validation record

The end-of-session validation recorded the following performance after adding the
history-replayed core and the downside-only checks:

| Cohort | Rows | MAE | RMSE | High-rated failures |
| --- | ---: | ---: | ---: | ---: |
| Fully refreshed validation rows | 12 | 148.0 | 217.1 | 0 of 6 outside +/-200 Elo |
| Reliable historical/readiness audit | 43 | 164.5 | 235.1 | 0 of 6 outside +/-200 Elo |

The original, pre-stratification stability cap corrected three known optimistic
estimates in the frozen June snapshot:

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

The normal scheduler runs the core and both automatic supplements without separate
commands:

```bash
python cli.py run -c config/benchmark.yaml -v
```

Before allowing calls, print the exact readiness-, freeze-, priority-, and
budget-aware plan with:

```bash
python cli.py run -c config/benchmark.yaml --api codex --acquisition-plan
```

This reads the same rating and game state as production but creates no model/API
calls and writes no benchmark results. It reports panels scheduled or deferred,
planned first-attempt calls, maximum retries, and estimated base cost.

For repair, isolated reruns, or research, the game-like panel can still be invoked
directly:

```bash
python position_benchmark/run_benchmark.py \
  --positions position_benchmark/panels/game_like_48.json \
  --output position_benchmark/results/game_like.json \
  --players "PLAYER_ID" \
  --depth 30 \
  --api codex
```

Existing rows predate conditional retry measurement. To refresh only selected rows
that lack the current retry-protocol stamp, add `--refresh-retry-evidence`. This
reruns the first attempt because a retry outcome is meaningful only when observed
conditionally after a fresh illegal answer; it does not manufacture a retry from a
stored historical illegal move. `--retry-missing` retains its narrower meaning of
filling absent position rows.

Likewise, the continuation probe can be invoked directly for repair or research:

```bash
python scripts/run_stability_probe.py \
  --players "PLAYER_ID" \
  --probe-plies 8 \
  --score-depth 30 \
  --api codex
```

The default `--limit 8` invokes the stratified selector. `--position-indices` is
an explicit research override and is recorded as such in the result summary.

For non-production validation, ratings can be recalculated from the same benchmark
means with a looser seed RD. This command is intentionally guarded: both options
are required, rating writes are forced to the specified local file, and
`data/ratings.json` is rejected as an output target. Game results may still be
read from the configured production store.

```bash
python cli.py recalculate \
  --validation-seed-rd 300 \
  --validation-output /tmp/chess-ratings-rd300.json
```

This counterfactual reduces, but does not eliminate, circularity because its prior
mean still comes from the position benchmark. Production remains fixed at RD 166.

Audit freshness, coverage, prediction error, and any available supplemental caps:

```bash
python scripts/audit_position_benchmark_readiness.py --show-all
```

The audit reads the separate automatic supplemental result files under
`position_benchmark/results/`
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
5. State the core and full automatic-acquisition call counts explicitly.
6. Preserve the model API/provider surface between the position benchmark, automatic
   probes, and rated games, or label the comparison as provider-mismatched.

Research scripts and newly generated panels are experiments until this change
control is completed.
