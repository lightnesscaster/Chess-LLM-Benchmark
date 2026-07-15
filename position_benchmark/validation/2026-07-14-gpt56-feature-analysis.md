# GPT-5.6 supplemental feature analysis — 2026-07-14

This report analyzes the 50-position core, 48-position game-like panel, new
`stratified-v2` continuation rows, and the initial 96 completed GPT-5.6 games. It made no new
model calls and did not change production ratings or prediction rules. All learned
models and alternate formulas below are research diagnostics only.

A subsequent 16-game targeted follow-up is recorded in
`validation/2026-07-14-gpt56-targeted-games.md`. Outcome-comparison metrics in
this report remain the frozen initial-96-game analysis; the retry audit and
legality table below have been refreshed to include all 112 games.

## Validation targets and uncertainty

Each GPT-5.6 configuration has eight games. Opponent-adjusted performance estimates
therefore have broad profile-likelihood intervals. The current full prediction is
clearly above the 95% interval for Luna medium and Terra low, and slightly above it
for Terra high. Most other apparent errors are not distinguishable from eight-game
sampling noise.

| Configuration | Game performance | 95% interval | Current prediction |
| --- | ---: | ---: | ---: |
| Luna medium | 229 | -294 to 585 | 1,064 |
| Terra low | 102 | -750 to 509 | 980 |
| Terra high | 483 | 134 to 880 | 943 |

The validation-only RD-300 recalculation, refreshed after the stratified results,
puts these configurations at 327, 297, and 595 respectively. Across all 12 models,
the current predictor has MAE 218 Elo and +97 Elo mean bias against that less
prior-dominated target. This target still shares the same game outcomes and is not
independent validation.

## Game-like panel

The 48-position panel contains signal, but its point rating is noisy and its four
categories are not equally informative.

- Raw aggregate game-like rating has Spearman correlation 0.692 with game
  performance.
- Defense CPL has the strongest relationship with the residual left by the core:
  Spearman -0.734. Tactical/equal is next at -0.364; advantage conversion is
  -0.210 and quiet/equal is -0.042.
- A defense-CPL downside diagnostic produced lower log loss (0.688) and lower
  opponent-adjusted MAE (263 Elo) than the current full predictor, but worse Brier
  score (0.173). It is based on only 12 defense positions and must not be promoted
  from this family-only result.
- Stratified position bootstrapping gives a mean game-like rating SD of about
  220 Elo. Several models' current 150-Elo cap decision activates in only 35–65%
  of resamples.
- Median, geometric-mean, trimmed-mean, and lower CPL winsorization all removed
  useful downside signal. The current 5000-CPL-capped arithmetic mean was better
  on log loss and game-performance MAE.

The current hard cap is also discontinuous: once the supplemental estimate crosses
the 150-Elo threshold, the prediction jumps down by at least 150 Elo. The continuous
deadband alternative `min(core, game_like + 150)` improved that behavior but was
worse on this cohort. A future replacement should propagate panel uncertainty
rather than select a point estimate with a hard threshold.

## Continuation quality

There is no stable evidence of general move-quality degradation over four model
moves.

- Model-weighted later-move minus first-move CPL is +16, with a model-cluster
  bootstrap interval of -51 to +98.
- Exact first-move agreement with the earlier game-like response is only 63.5%
  (bootstrap interval 53.1–72.9%).
- After rescoring both answers at Stockfish depth 10, first-move CPL correlation is
  only 0.217. In 20.8% of repeats, CPL differs by at least 100; large better/worse
  flips occur in both directions. This is one-shot measurement variance, not a
  dependable downside feature.

Later-ply CPL is substantially confounded by the seeded random opponent:

- Random replies average 365 CPL with median 106 CPL.
- 22.7% lose at least 300 CPL and 4.3% lose at least 1000 CPL.
- The mean absolute evaluation before the model moves rises from 320 CPL on the
  first move to 1,044 CPL on the third.
- Random-reply CPL correlates 0.400 with the following model CPL.

The random opponent is useful for exercising legal-move retries and exposing the
model to unusual boards. It is not a sound opponent for measuring representative
continuation chess quality. Later-ply CPL and catastrophe counts should therefore
not be promoted as rating evidence without a deterministic, legal chess opponent.

## Legality and the retry protocol

This is the strongest finding, with an important post-analysis correction.
Production uses two illegal strikes across the entire game. The first illegal gets
a retry; if that succeeds, a later second illegal response forfeits immediately
without another retry. The original analysis incorrectly described this as an
independent retry on every turn.

| Configuration | Core legal | Game-like legal | Continuation legal | First-attempt illegal rate in games | Failed same-turn retries | Later second strikes | Game forfeits |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Luna medium | 98.0% | 97.9% | 97.0% | 6.30% | 1/14 | 10 | 11/16 |
| Terra low | 96.0% | 100.0% | 97.0% | 3.94% | 0/11 | 8 | 8/16 |
| Sol high | 98.0% | 100.0% | 97.0% | 0.96% | 0/3 | 0 | 0/8 |

The refreshed ply-aware audit found 66 retry opportunities across all 112 GPT-5.6
games: 64 recovered, two failed, and none were unknown. Twenty-seven forfeiting
second strikes occurred on later turns after a successful retry. Retry failure was
therefore not the dominant GPT-5.6 problem; repeated first-attempt illegality in
game contexts was. The machine-readable audit is
`validation/2026-07-14-gpt56-game-retry-audit.json`.

The correct protocol model needs two quantities:

- `p`: probability that the first response on a turn is illegal;
- `q`: probability that the retry is also illegal, conditional on an initial
  illegal response.

Under the actual policy, the probability of surviving `T` turns is
`(1-p)^T + T*p*(1-p)^(T-1)*(1-q)`. The current production feature is the `q=0`
special case. `(1-p*q)^T` would describe a different policy that grants a fresh
retry on every turn. Existing static rows estimate `p`; they do not yet estimate
model-specific `q`.

## Prediction comparisons

Metrics below use all 96 saved outcomes. Opponent-adjusted MAE uses the noisy
eight-game point estimates and should be interpreted alongside outcome metrics.

| Predictor | Brier | Log loss | MAE | Mean bias |
| --- | ---: | ---: | ---: | ---: |
| Core | 0.1642 | 0.7766 | 343 | +219 |
| Core + raw game-like hard cap | 0.1696 | 0.7432 | 294 | +141 |
| Current full stratified predictor | 0.1675 | 0.7098 | 277 | +122 |
| Direct minimum, no 150-Elo trigger (research) | 0.1665 | 0.7006 | 268 | +113 |
| Defense-CPL diagnostic (research) | 0.1729 | 0.6879 | 263 | +56 |

Relative to core, the current full predictor improves log loss by 0.067; the
model-cluster bootstrap interval is 0.012 to 0.142 and 99.3% of resamples improve.
It worsens Brier by 0.0033, with an interval from -0.0099 improvement to +0.0208
worsening. The supplement reduces extreme overconfidence but does not improve every
notion of point accuracy.

Leave-one-base-model-out ridge diagnostics were deliberately grouped by Sol,
Terra, and Luna. Core + game-like + defense produced MAE 414; adding continuation
features produced MAE 415. Both are worse than the fixed core and fixed current
predictors. Twelve same-generation variants are insufficient to learn a reliable
new regression, and no fitted candidate should be used in production.

## Conclusions and next changes to consider

1. Keep the 48-position game-like panel. Preserve all four categories and report
   category metrics separately; defense is the most promising residual diagnostic.
2. Treat the current random-opponent continuation panel as a protocol/legality
   stress test, not a chess-quality rating. Do not rely on later-ply CPL until the
   opponent is replaced with a deterministic legal chess policy.
3. Keep conditional retry testing in static benchmark rows, but prioritize better
   measurement of repeated first-attempt illegality. The six-position frozen
   legality-stress candidate raises held-out GPT-5.6 illegal incidence 2.78x and
   can collect both signals with far fewer ordinary calls. Its three-model pilot
   reproduced 2/18 aggregate illegals and recovered both retries, but the illegal
   model-position pairs did not repeat; use it as a model-level stress sample, not
   a deterministic trap list.
4. Refit the legality survival feature with the two-strikes-per-game `p, q` model
   once retry evidence exists. Do not substitute the new feature into coefficients
   learned with the old proxy. Keep first-attempt illegality and failed-retry risk
   as separate diagnostics until then.
5. Replace the hard 150-Elo cap eventually with an uncertainty-aware, continuous
   combination. Do not select its form from these 12 variants alone.
6. Do not change production from this analysis yet. The measurement diagnosis is
   strong, but choosing new coefficients or promoting defense weights still needs
   evidence outside GPT-5.6. No cross-family probes were run for this report.
