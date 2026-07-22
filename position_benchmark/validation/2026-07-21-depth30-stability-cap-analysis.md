# Depth-30 continuation-cap audit — 2026-07-21

This zero-call audit evaluates the corrected depth-30 continuation artifacts. It
did not write the production rating store. The accompanying predictor change only
deduplicates catastrophe events within each trajectory. The cohort has
16 configurations across
7 model-line families.

## Result

No hard-cap redesign is validated for production. The two game-rating targets
disagree: removing the hard cap helps the independent no-position-seed target but
slightly hurts the higher-RD position-seeded target. Neither direction is robust
under family resampling and rating uncertainty. The inherited evidence gate needs
at least 30 configurations and
8 families; both coverage checks fail.

The current move-level catastrophe count is structurally wrong for an absorbing
loss: repeated losing moves in one continuation are correlated and must not be
treated as independent catastrophe events. `deduplicated_move_exposure_cap`
implements a narrow correction: retain at most the first catastrophe in each
trajectory, preserve the existing move-exposure denominator and coefficients,
and therefore never make a cap harsher. This correction fits no target data.
`trajectory_hazard_cap` additionally censors later exposures; that more ambitious
survival-style redesign remains diagnostic rather than validated rating evidence.

The production change is limited to deduplicating catastrophes within each
trajectory. Existing cap constants, the 150-Elo deadband, continuation legality,
and forfeit evidence remain unchanged. Any coefficient refit, complete CPL-cap
removal, or survival-hazard redesign remains blocked until a newly frozen cohort
passes the evidence gate.

## Current depth-30 comparison

### Position-seeded validation ratings with RD 300

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 287.3 | 396.2 | +64.1 | 531.6 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 292.8 | 398.2 | +82.4 | 537.1 | 0.000 | 0.347 |
| `trajectory_hazard_cap` | 293.1 | 398.6 | +82.1 | 537.6 | 0.000 | 0.340 |
| `two_affected_trajectory_gate` | 292.8 | 398.2 | +82.4 | 537.1 | 0.000 | 0.345 |
| `repeated_forfeit_only` | 226.2 | 297.7 | +177.3 | 235.0 | 0.750 | 1.000 |
| `no_hard_cap` | 240.7 | 317.3 | +191.8 | 260.2 | 0.677 | 0.991 |
### No-position-seed game ratings

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 339.3 | 427.6 | +22.6 | 566.2 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 321.1 | 416.0 | +40.9 | 561.2 | 0.662 | 0.888 |
| `trajectory_hazard_cap` | 321.4 | 416.4 | +40.5 | 561.7 | 0.659 | 0.884 |
| `two_affected_trajectory_gate` | 321.1 | 416.0 | +40.9 | 561.2 | 0.660 | 0.891 |
| `repeated_forfeit_only` | 253.5 | 320.2 | +135.7 | 255.1 | 0.978 | 1.000 |
| `no_hard_cap` | 268.1 | 338.5 | +150.3 | 279.8 | 0.875 | 0.998 |

All bootstrap and uncertainty probabilities compare the candidate against
`current_move_cap`. Candidates are fixed structural alternatives; no coefficients
were fitted to these targets.

## Rows whose rating is currently changed by the hard cap

| Model | Catastrophic moves | Affected catastrophe starts | All affected starts | Current | Deduplicated | Hazard-style | No hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-3.1-flash-lite-preview | 0 | 0 | 2 | 500 | 500 | 500 | 732 |
| gpt-5.6-luna (high) | 2 | 2 | 2 | 598 | 598 | 595 | 1090 |
| deepseek-v4-flash (max) | 3 | 1 | 1 | 575 | 867 | 867 | 867 |
| gpt-3.5-turbo-instruct | 2 | 2 | 2 | 600 | 600 | 598 | 1627 |

## Historical sensitivity

The stale June depth-10/pre-stratification snapshot is not current validation, but
it explains why the cap existed. Its three-row MAE was
107.3 with the
old cap, 222.7
with repeated-forfeit-only protection, and
399.3 with no hard cap.
Thus repeated-forfeit protection retains much of the historical weak-model value,
while the present cohort does not validate a direct random-reply CPL cap.

## Decision boundary

- Do not fit new cap coefficients on 16 configurations.
- Do not describe RD-300 as independent; it still shares the benchmark prior.
- Production may deduplicate catastrophes within a trajectory because this cannot
  increase any penalty and does not fit the validation targets.
- Do not deploy hazard censoring, new coefficients, or cap removal on this cohort.
- Re-run this fixed audit automatically as current depth-30 supplements accumulate.
