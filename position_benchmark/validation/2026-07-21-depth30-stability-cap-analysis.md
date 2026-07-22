# Depth-30 continuation-cap audit — 2026-07-21

This zero-call audit evaluates the corrected depth-30 continuation artifacts. It
did not write the production rating store. The accompanying predictor change only
deduplicates catastrophe events within each trajectory. The cohort has
14 configurations across
5 model-line families.

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
| `current_move_cap` | 232.1 | 307.4 | +126.9 | 320.1 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 238.4 | 310.4 | +147.8 | 329.2 | 0.000 | 0.352 |
| `trajectory_hazard_cap` | 238.7 | 310.6 | +147.5 | 329.3 | 0.000 | 0.345 |
| `two_affected_trajectory_gate` | 238.4 | 310.4 | +147.8 | 329.2 | 0.000 | 0.347 |
| `repeated_forfeit_only` | 235.6 | 308.1 | +182.9 | 328.0 | 0.274 | 0.472 |
| `no_hard_cap` | 235.6 | 308.1 | +182.9 | 328.0 | 0.269 | 0.472 |
### No-position-seed game ratings

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 291.3 | 351.7 | +79.3 | 372.5 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 270.5 | 335.5 | +100.2 | 364.1 | 0.676 | 0.895 |
| `trajectory_hazard_cap` | 270.7 | 335.7 | +100.0 | 364.2 | 0.672 | 0.887 |
| `two_affected_trajectory_gate` | 270.5 | 335.5 | +100.2 | 364.1 | 0.669 | 0.893 |
| `repeated_forfeit_only` | 266.6 | 332.6 | +135.3 | 362.2 | 0.917 | 0.736 |
| `no_hard_cap` | 266.6 | 332.6 | +135.3 | 362.2 | 0.923 | 0.736 |

All bootstrap and uncertainty probabilities compare the candidate against
`current_move_cap`. Candidates are fixed structural alternatives; no coefficients
were fitted to these targets.

## Rows whose rating is currently changed by the hard cap

| Model | Catastrophic moves | Affected catastrophe starts | All affected starts | Current | Deduplicated | Hazard-style | No hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gpt-5.6-luna (high) | 2 | 2 | 2 | 598 | 598 | 595 | 1090 |
| deepseek-v4-flash (max) | 3 | 1 | 1 | 575 | 867 | 867 | 867 |

## Historical sensitivity

The stale June depth-10/pre-stratification snapshot is not current validation, but
it explains why the cap existed. Its three-row MAE was
107.3 with the
old cap, 197.7
with repeated-forfeit-only protection, and
399.3 with no hard cap.
Thus repeated-forfeit protection retains much of the historical weak-model value,
while the present cohort does not validate a direct random-reply CPL cap.

## Decision boundary

- Do not fit new cap coefficients on 14 configurations.
- Do not describe RD-300 as independent; it still shares the benchmark prior.
- Production may deduplicate catastrophes within a trajectory because this cannot
  increase any penalty and does not fit the validation targets.
- Do not deploy hazard censoring, new coefficients, or cap removal on this cohort.
- Re-run this fixed audit automatically as current depth-30 supplements accumulate.
