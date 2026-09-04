# Legacy Position-Benchmark Artifacts

These files are retained for provenance and numeric-index compatibility. They are
not active benchmark inputs.

- `combined_positions_75.json`: the former 25-blunder + 50-equal registry. Its
  indices are preserved as `legacy_position_idx` in the organized panel files.
- `combined_results_75.json`: the pre-migration mixed result file.
- `game_like_results_pre_stable_ids.json`: game-like results before stable IDs.
- `equal_results_pre_unified.json` and `survival_results.json`: older research data.

Use `../benchmark_manifest.json` to locate every active panel and result file.
The benchmark runner rejects the combined legacy registry unless historical work
explicitly supplies `--allow-legacy-input`.
