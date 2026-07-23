# Depth-30 continuation-cap audit — 2026-07-21

This zero-call audit evaluates the corrected depth-30 continuation artifacts. It
did not write the production rating store. The accompanying predictor change only
deduplicates catastrophe events within each trajectory. The cohort has
28 configurations across
16 model-line families.

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
| `current_move_cap` | 214.9 | 314.7 | +65.5 | 213.7 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 224.3 | 318.6 | +82.2 | 231.0 | 0.000 | 0.132 |
| `trajectory_hazard_cap` | 223.9 | 318.8 | +81.4 | 230.2 | 0.081 | 0.136 |
| `two_affected_trajectory_gate` | 218.0 | 316.2 | +75.9 | 215.0 | 0.000 | 0.346 |
| `repeated_forfeit_only` | 193.3 | 251.8 | +143.5 | 170.0 | 0.643 | 0.944 |
| `no_hard_cap` | 201.6 | 265.1 | +151.8 | 175.9 | 0.599 | 0.821 |
### No-position-seed game ratings

| Candidate | MAE | RMSE | Bias | IVW MAE | Family bootstrap P(improves) | RD simulation P(improves) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `current_move_cap` | 273.3 | 373.1 | +10.0 | 219.7 | 0.000 | 0.000 |
| `deduplicated_move_exposure_cap` | 269.2 | 367.6 | +26.8 | 234.8 | 0.542 | 0.807 |
| `trajectory_hazard_cap` | 268.7 | 367.8 | +25.9 | 234.1 | 0.632 | 0.802 |
| `two_affected_trajectory_gate` | 262.9 | 365.5 | +20.5 | 218.6 | 0.643 | 0.891 |
| `repeated_forfeit_only` | 237.6 | 311.1 | +88.1 | 174.3 | 0.803 | 0.992 |
| `no_hard_cap` | 245.9 | 322.1 | +96.3 | 179.9 | 0.721 | 0.971 |

All bootstrap and uncertainty probabilities compare the candidate against
`current_move_cap`. Candidates are fixed structural alternatives; no coefficients
were fitted to these targets.

## Rows whose rating is currently changed by the hard cap

| Model | Catastrophic moves | Affected catastrophe starts | All affected starts | Current | Deduplicated | Hazard-style | No hard cap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| gemini-2.5-flash (no thinking) | 4 | 2 | 3 | 461 | 637 | 637 | 637 |
| gpt-5.1 (no thinking) | 4 | 4 | 4 | 550 | 550 | 531 | 747 |
| gpt-5.6-luna (high) | 2 | 2 | 2 | 598 | 598 | 595 | 1090 |
| deepseek-v4-flash (max) | 3 | 1 | 1 | 575 | 867 | 867 | 867 |
| gemini-3.1-flash-lite-preview | 0 | 0 | 2 | 500 | 500 | 500 | 732 |
| gpt-3.5-turbo-instruct | 2 | 2 | 2 | 600 | 600 | 598 | 1627 |

## Historical sensitivity

The stale June depth-10/pre-stratification snapshot is not current validation, but
it explains why the cap existed. Its three-row MAE was
107.3 with the
old cap, 298.7
with repeated-forfeit-only protection, and
399.3 with no hard cap.
Thus repeated-forfeit protection retains much of the historical weak-model value,
while the present cohort does not validate a direct random-reply CPL cap.

## Decision boundary

- Do not fit new cap coefficients on 28 configurations.
- Do not describe RD-300 as independent; it still shares the benchmark prior.
- Production may deduplicate catastrophes within a trajectory because this cannot
  increase any penalty and does not fit the validation targets.
- Do not deploy hazard censoring, new coefficients, or cap removal on this cohort.
- Re-run this fixed audit automatically as current depth-30 supplements accumulate.
