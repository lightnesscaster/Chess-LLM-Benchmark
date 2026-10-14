# Benchmark Seed RD Backtest Plan

**Goal:** Compare starting RDs 166, 250, 300, and 350 against subsequent game outcomes before changing rating policy.

**Architecture:** A read-only completion-order replay scores each model's expected score before applying that result to either player. Primary evidence uses benchmark predictions locked before first recorded result; an explicitly retrospective sensitivity run uses current frozen benchmark predictions. No published ratings or results are written by the evaluator.

**Tech stack:** Python, existing Glicko-2 implementation, pytest, JSON snapshots.

**Spec:** Settled requirements are captured below.

## Global constraints

- Use the existing checkout; preserve unrelated local changes and ongoing acquisition jobs.
- Freeze results, benchmark predictions, shadow locks, engine configuration and model metadata with SHA-256 hashes under private `data/seed_rd_backtest_20261014/`.
- Candidates are fixed at 166, 250, 300, 350 before examining outcome metrics.
- Use one chronological pass, simultaneous player updates, historical anchor values, fixed human snapshots, and production ghost exclusions. Never use final ratings or future legality statistics. Results lack start timestamps: disclose that forecasts are pre-result, and cannot reconstruct actual start-time ratings for overlapping games.
- Forecast each seeded player's score with production Glicko expected score, using opponent RD. Draws score 0.5. Report squared expected-score error (Brier), fractional-score cross entropy, per-family macro averages and first-ten-game results. A match between two seeded models contributes two explicitly identified player observations, not two independent games.
- Prospective benchmark means must have a lock timestamp earlier than the player's first recorded result and a zero-game eligibility snapshot. Non-cohort opponents use benchmark-free chronological ratings, a conservative limitation to document.
- Compare results excluding Argon as a sensitivity check, plus per-family consistency. Do not call this untouched confirmation: the proposal was motivated by Argon's known results.
- Historical fixed benchmark means are a secondary retrospective audit; their missing timestamps and predictor training prevent claims of fully held-out validation.
- Change the global default only if prospective and historical checks support a consistent improvement across families. Otherwise retain 166 and document why. Any production migration requires a separate reviewed, backed-up preview and must not republish games.

## Task 1: Tested chronological evaluator

Files: `rating/seed_rd_backtest.py`, `tests/test_seed_rd_backtest.py`.

- [x] Write failing tests for pre-update scoring, prefix invariance, simultaneous updates, dated seeds, historical anchors, human snapshots, ghosts, invalid/duplicate games and family aggregation.
- [x] Run the tests and confirm they fail because the evaluator is missing.
- [x] Implement `replay(results, seeds, engines, model_ids, seed_rd)` returning forecast observations, final ratings and rejected-result reasons, using local in-memory state only. Seeds contain `rating`, `available_at`, `family`, and `lab`.
- [x] Run focused tests and existing Glicko/anchor/seed tests.

## Task 2: Frozen audit and decision

Files: `scripts/backtest_seed_rd.py`, `rating/validation/2026-10-14-seed-rd.md`.

- [x] Freeze inputs and hashes; expose only sanitized aggregate outputs in the committed report.
- [x] Replay each RD against the prospective cohort and retrospective benchmark snapshot. Report full/first-ten-game and with/without-Argon metrics plus per-family rows; do not report an individual game's human identity.
- [x] Request independent review of methodology, code and recommendation; resolve material findings and rerun checks.
- [x] Record exact commands, metrics, sample limitations and a global-policy recommendation. If no sufficiently consistent evidence exists, explicitly retain production RD166 without rerating any games.
- [x] Conditional migration: not triggered. Primary evidence is inconsistent outside Argon; retain166 and perform no production writes.

Verification: 36 tests and two subtests passed. Frozen data and implementation hashes are recorded in the private audit report. See `rating/validation/2026-10-14-seed-rd.md` for the decision and timing limitations.
