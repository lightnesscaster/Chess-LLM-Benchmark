# Durable engine-clock checkpoints

Goal: preserve a saved match's selected engine time control and remaining clocks across process interruptions, with a tested official-CLI resume path.

## Decisions and constraints

- Work in the current checkout; leave existing results, private credentials and running matches alone. This applies to newly started games using the updated runner.
- Saved manual games and scheduler games write private checkpoints under `data/checkpoints/`; no-save tests/probes do not write checkpoints. Checkpoints contain no API keys or raw prompts/responses and are never part of the public live-game schema.
- Add `UCIEngine.snapshot_clock() -> dict | None` and `restore_clock(state: dict) -> None`. Clock snapshots are versioned and contain exact scaled white/black remaining seconds, scaled increment, selected nominal time control, speed scale, measured/reference NPS, and a deterministic configuration fingerprint. Initialize a deferred clock once when a checkpoint is first taken. Restore validates everything before mutation and never re-probes or re-draws the clock.
- Atomic checkpoint writes use a same-directory temporary file, mode 0600, fsync and replace. An OS file lock prevents concurrent active resumptions. Preserve UUID, complete legal move history/FEN, player identities/configuration, move limit, strikes/retry counters, paid token/timing accounting, pending draw negotiations, frozen rating context and engine clocks.
- `GameRunner` adds optional `checkpoint_dir=None` and `resume_checkpoint=None` path parameters. Resume validates schema, identities and replay/FEN consistency before any provider/engine move call, restores the saved UUID, and resumes the exact side to move. Completed checkpoints return their saved result/PGN without making another move; normal saved-result publication remains UUID-idempotent.
- Save before a request, after illegal responses and after every committed move. A resumed in-flight LLM request marks accounting partial because an unreported charge may have occurred. Preserve pending retry context and never grant a fresh strike allowance.
- During a clocked engine search, run the blocking search off the event loop and checkpoint its elapsed thinking time at one-second intervals without crediting an increment before a move is committed. Resume uses the last durable clock value, with at most one heartbeat interval of unrecorded search time; offline downtime is not charged.
- `cli.py manual --resume-checkpoint PATH` uses the original matchup flags, requires one game, rejects `--moves`, and runs through the official save/publication flow. Saved manual games enable checkpointing automatically; local-only saved games get the same protection. A clocked-engine `--moves` continuation without a clock checkpoint must fail before play instead of silently giving it a fresh clock.
- Existing `--moves` remains available for LLM-only continuations. Do not modify production ratings or republish unrelated games.

## Tasks and ownership

1. Engine clock serialization: `engines/uci_engine.py`, focused `tests/test_uci_engine.py` / new clock tests. TDD exact roundtrip, deferred initialization, no redraw/probe, configuration mismatch, malformed/nonfinite values and genuine UCI limits after restoration.
2. Checkpoint storage and runner integration: `game/checkpoints.py`, `game/game_runner.py`, new checkpoint tests. TDD interruption/resume, partial clock heartbeat, locking/atomicity, mismatched identities/history, strikes/retries/draw/accounting continuity and completed-result no-replay behavior.
3. Official entrypoints: `cli.py`, `game/match_scheduler.py`, README and focused CLI/scheduler tests. Wire defaults and strict resume flags, with a no-cost official-CLI smoke.
4. Independent review of each boundary and the integrated patch; run full Python/JS checks appropriate to these changes. Commit only code/docs/tests after verification; retain checkpoint artifacts locally.
