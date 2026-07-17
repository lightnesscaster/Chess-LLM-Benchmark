# Frozen supplemental predictor evaluation plan — 2026-07-17

This zero-model-call analysis tests whether the automatically acquired game-like,
continuation, first-answer illegality, and retry evidence improve rating
prediction beyond the fixed production predictor.

The game target is an isolated recalculation with the current position prediction
as prior mean and RD 300. Production continues to use RD 166 and cannot be written
by the validation command. The target is less prior-dominated, but it is not
independent because the prior mean still comes from the benchmark.

## Leakage control

Every learned candidate uses outer leave-one-model-line-family-out validation.
Luna, Terra, and Sol are separate GPT-5.6 base-model families; reasoning levels
inside each line stay together. Ridge strength is selected only by inner
family-held-out validation within the outer training fold.

The analysis uses separate game-like, continuation, and all-panel cohorts because
requiring every supplement would discard usable cross-family evidence. Cohort
membership follows fixed readiness and game-evidence rules recorded in the JSON
plan.

## Decision boundary

No result can change production directly. A candidate can only earn a recommendation
for new independent validation if it has at least 30 configurations and eight
families, improves MAE by at least 10% and RMSE by at least 5%, wins in at least
half of families, worsens no family by more than 100 Elo, and has at least 90%
family-cluster bootstrap probability of improving MAE.

The exact input hashes, feature sets, metrics, alpha grid, cohort rules, and output
paths are frozen in `2026-07-17-supplemental-predictor-plan.json`.

Before the first successful analyzer run, the plan was amended to add omitted
historical-blunder hashes required to reproduce the fixed production baseline and
to state the already-planned beta-binomial retry prior explicitly. This occurred
after the isolated target refresh; no cohort rule, candidate, metric, threshold,
or output changed.
