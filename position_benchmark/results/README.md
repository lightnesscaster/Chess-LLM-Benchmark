# Panel Results

Results are separated by the panel that generated them:

- `core.json` corresponds only to `../panels/core_equal_50.json`.
- `game_like.json` corresponds only to `../panels/game_like_48.json`.
- `optional_blunder.json` corresponds only to
  `../panels/optional_blunder_25.json`.
- `stability.json` contains continuation-probe summaries/results.

New result rows use `position_id` as the stable key and retain panel-local
`position_idx` for ordering. They are stamped `result_schema_version: 2`.
`legacy_position_idx` exists only for compatibility.
The optional blunder results intentionally omit 361 historical rows whose FEN did
not match their claimed old numeric index.
