# Corrected CPL conversion validation

> For agentic workers: use superpowers:subagent-driven-development task by task, with independent review.

**Goal:** Revalidate the production position-to-rating conversion on corrected CPL scores, recalibrate only when held-out evidence supports it, and publish any justified change with a guarded rating replay.

**Architecture:** Freeze production inputs and derive game-only targets with benchmark seeds disabled. Evaluate a small, predeclared candidate set with nested family-held-out validation. Keep Carbon out of fitting and selection. Separately validate seed uncertainty. Any production change must be versioned, tested, audited, backed up and verified against live APIs.

**Tech stack:** Python, NumPy/SciPy, existing rating replay, Firestore, pytest.

**Spec:** User authorization in this chat; the frozen statistical policy is `position_benchmark/validation/2026-11-11-corrected-cpl-calibration-plan.json`.

## Global constraints

- Existing checkout; preserve unrelated edits. No new model calls, games, position rescoring, anchor changes or acquisition-policy changes.
- Use only current `root-reference-v1` production panels, preserving optional blunder inputs and supplemental caps.
- Targets are an explicit saved-game snapshot replayed with **all benchmark seeds disabled**, initial RD 350 and the unchanged game/anchor rules. Published seeded ratings are not training targets.
- Primary development cohort: at least 8 games and game-only RD at most 200, excluding Carbon. Reliability sensitivity: at least 20 games/RD at most 150; mature sensitivity: at least 30 games/RD at most 120. Carbon is diagnostic only.
- Primary grouping follows the existing model-line prefixes, with explicit new Astra, Sol 6.1, Argon and Carbon lines. Reasoning variants never cross train/test boundaries. Provider-held-out validation is a stress check.
- Never rewrite historical calibration artifacts or tune thresholds after inspecting outcomes.

## Tasks

- [ ] **1. Freeze data and create targets (parent).** Save configuration, complete production game/rating/benchmark snapshots and hashes privately under `data/calibration_20261111/`. Replay using `recalculate_ratings` in validation mode, `validation_disable_benchmark_seeds=True`, explicit games and empty prediction map. Save per-model features/targets in a reproducible non-private matrix. Check game counts, no production writes, current scorer identity and source hashes.
- [ ] **2. Implement and test validation (worker).** Own `position_benchmark/calibration_validation.py`, `scripts/validate_corrected_calibration.py` and focused tests. Compare fixed current predictor; full-predictor intercept correction; nonnegative affine correction; and monotone three-feature core refits with ridge alpha 0, 1, 10 or 100 toward the old slopes. Preserve the production panel/cap composition. Select only inside each outer training fold using inner leave-family-out macro-family MAE, then RMSE and deterministic simplicity ties. Evaluate the complete selection procedure against fixed production predictions on outer held-out families and held-out providers. Family bootstrap: 5,000, seed 1729.
- [ ] **3. Independent methodology/results review.** Verify target independence, grouping, no Carbon leakage, training-only standardization/selection, complete cap equivalence, uncertainty calculation, promotion gates, hashes and reproducibility. Generate JSON and a readable report. If evidence fails, retain the relevant production component and explain why.
- [ ] **4. Conditional implementation and rollout (parent/worker).** If the frozen gate passes, integrate the chosen conversion as an explicit calibration version. Widen seed RD only under its separate gate. Ensure web/CLI/publication use identical means and uncertainty. Add regression tests, full tests, documentation and a guarded production replay with backup and unchanged game records. Deploy and verify live values. No approval repetition is needed for this already-authorized recalibration.

## Review and execution ledger

- Ruling: this is a validation-and-conditional-rollout task already authorized by the user. Statistical choices below are implementation decisions; production remains unchanged until validation succeeds.
- Ruling: compare low-dimensional recalibration with the original-feature refit, rather than expanding the benchmark feature set. The user requested calibration of existing evidence.
- Ruling: preserve the current cap mechanics during fitting; an affine correction calibrates the complete current predictor. Acquisition and game protocols remain unchanged.
- Ruling: use conservative held-out predictive RMSE for uncertainty, including target noise; do not increase confidence by reducing RD in this audit.
