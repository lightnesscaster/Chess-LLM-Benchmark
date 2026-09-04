# Selected Benchmark Panels

These are the only selected position sets used by the production predictor:

- `core_equal_50.json` — required 50-call production benchmark.
- `game_like_48.json` — optional downside-only diagnostic.
- `optional_blunder_25.json` — optional historical diagnostic with little measured
  incremental value.

Each panel has its own local `position_idx` space beginning at zero. `position_id`
is the stable cross-file key. Do not concatenate panels or describe their combined
count as the benchmark size.
