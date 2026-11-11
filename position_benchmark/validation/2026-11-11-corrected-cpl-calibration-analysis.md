# corrected-cpl-calibration-20261111-v1

Eligible configurations: 40; families: 21.
Carbon is diagnostic only. Targets use the benchmark-seed-free game replay.

| Method | Row MAE | Row RMSE | Macro MAE | Macro RMSE | Bias | Max error |
|---|---:|---:|---:|---:|---:|---:|
| Fixed production | 224.231 | 281.365 | 210.814 | 259.670 | -97.578 | 833.870 |
| Nested selected | 225.022 | 274.762 | 215.172 | 266.477 | -1.977 | 715.324 |
| Provider held out | 218.505 | 271.481 | 198.490 | 245.616 | 26.368 | 810.443 |

Mean promotion: FAIL.
Paired bootstrap MAE improvement probability: 0.4422.

- minimum_configurations: PASS
- minimum_families: PASS
- macro_mae_improvement: FAIL
- macro_rmse_improvement: FAIL
- bootstrap_probability: FAIL
- family_win_fraction: PASS
- maximum_family_worsening: FAIL
- reliable_subset: PASS
- mature_subset: PASS
- provider_stress: PASS

Sensitivity and provider stress metric ratios (maximum 1.05):
- reliable (25 configurations): row_mae=0.9737; row_rmse=0.9858; macro_mae=0.9827; macro_rmse=1.0154
- mature (19 configurations): row_mae=0.9368; row_rmse=0.9788; macro_mae=0.9442; macro_rmse=1.0064
- provider stress: row_mae=0.9745; row_rmse=0.9649; macro_mae=0.9415; macro_rmse=0.9459

Full-cohort candidate after validation: `affine`.
Production mean action: **retain**.
Uncertainty uses **fixed** outer family predictions; lower bootstrap RMSE: 217.913.
Seed RD recommendation: **300** (previous 166).

No forced refit. Only components passing their separate frozen gates may be published.
Complete fold membership, inner selection scores, normalization, coefficients, sensitivity subsets, and hashes are in the JSON artifact.


Tested environment: Python 3.14.2, NumPy 2.5.4, SciPy 1.18.1.
- Limitation: Retrospective validation on the existing game graph, not independent prospective evidence.
- Limitation: Incumbent formula/cap development may overlap historical model cohorts; family holdout excludes families from this audit refit only.
- Limitation: Keep the separate frozen prospective stability-cap study unchanged.