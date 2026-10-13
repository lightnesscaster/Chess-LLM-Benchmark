# Live benchmark games implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Let visitors watch saved benchmark games while the LLMs play.

**Architecture:** Opt-in GameRunner broadcasts publish sanitized per-game snapshots through a background worker to the existing Firestore backend, with local JSON snapshots for local development. A public `/live` page polls a short-cached read-only `/api/live-games` endpoint and displays current boards and move history. Terminal snapshots remain visible briefly and link to the normal completed-game replay.

**Tech stack:** Existing Python, asyncio, Flask, Firestore, chess.js, Chessboard.js, and plain JavaScript; no new dependencies.

**Spec:** None — settled requirements are captured below.

## Global constraints

- Work in the existing checkout on `codex/live-games`; preserve unrelated changes and the running Astra–Argon process.
- Public snapshots contain player IDs, board FEN, UCI/SAN move lists, turn, timestamps, status, illegal counters and result only. Never publish prompts, raw responses, secrets, or private human game fields.
- Publication is best effort and must not block model inference or alter outcomes, PGN, ratings, or completed-game publication.
- Manual `--no-save` and `--local-only` games and position/stability probes do not broadcast. Scheduled saved benchmark games broadcast.
- Distinct game IDs prevent concurrent games overwriting each other.
- Heartbeats keep slow thinking visible. After 90 seconds without a heartbeat, show `interrupted`, rather than incorrectly implying a model is still thinking. Active/interrupted snapshots expire after 30 minutes; terminal snapshots expire after 10 minutes.
- API responses use `Cache-Control: no-store`; process-local backend caching lasts at most 2 seconds. Storage outages return HTTP 503, distinguishable from an empty live list.
- UI polls every 4 seconds, avoids overlapping fetches, pauses polling while hidden, and distinguishes live, finished, interrupted, loading, empty, and connection-error states.
- The default view follows the latest move. Viewing earlier moves is preserved through updates; a Go live button resumes following. Use textContent for model names and SAN text.
- Match the existing navy palette (`#1a1a2e`, `#16213e`, `#0f3460`, `#e94560`, `#eaeaea`) and system sans typography. Make the chessboard the main visual; desktop has board plus game selector/move list, mobile stacks them. Keyboard controls and visible focus are required; avoid decorative animation.

## Task 1: Snapshot publishing and public API

**Files:** Create `game/live_games.py`, `tests/test_live_games.py`, `tests/test_web_live_games.py`; modify `game/game_runner.py`, `game/match_scheduler.py`, `cli.py`, `web/app.py`, `README.md`, and focused runner/manual tests.

**Interfaces:**
- `LiveGameStore.list_games(now: float | None = None) -> list[dict]` returns sanitized snapshots, sorted active-first/newest-first, filtered by expiry and with stale active records converted to interrupted.
- Snapshot fields: `game_id`, `white`, `black`, `fen`, `moves` (UCI list), `san_moves`, `turn` (`white`/`black`), `status` (`active`/`finished`/`error`/`interrupted`), `started_at`, `updated_at` (ISO UTC), `heartbeat_at` and `expires_at` (Unix seconds), `winner` (`white`/`black`/`draw`/null), `termination`, `illegal_moves_white`, `illegal_moves_black`. `result` uses `*`, `1-0`, `0-1`, `1/2-1/2`.
- `GameRunner(..., publish_live: bool = False)` opts into publishing. Constructor/default must remain compatible with existing tests and position probes.
- `GET /api/live-games` returns `{ "games": [snapshots] }`; unavailable storage returns `{ "error": "Live games are temporarily unavailable." }` with 503.
- `GET /live` renders `live.html`; it does not load the leaderboard or historical games.

- [x] Write failing tests proving concurrent games remain separate, snapshots exclude private fields, heartbeats refresh expiry, stale games show interrupted, terminal states expire, publishing exceptions do not change a legal game, and disabled broadcasting performs no storage I/O.
- [x] Run `.venv/bin/python -m pytest tests/test_live_games.py tests/test_web_live_games.py -q` and inspect expected failures before implementation.
- [x] Implement snapshot building by replaying `board.move_stack` from its root, preserving promotion and castling SAN. Use atomic local writes and per-ID Firestore writes. Coalesce queued snapshots and publish off the asyncio event loop with bounded storage timeouts.
- [x] Integrate initialization, accepted moves, terminal states, and cancellation cleanup in GameRunner. Heartbeat every 15 seconds. Finish/cancel with bounded best-effort flushing; do not modify ratings.
- [x] Enable broadcasting only for saved production manual games and scheduler games. Add documented backend behavior.
- [x] Add endpoint/page tests for anonymous reads, expiry, unavailable backend, no-store cache headers and arbitrary model names; run the relevant runner/publication regressions.

## Task 2: Live viewer

**Files:** Create `web/templates/live.html`, `web/static/js/live.js`, `tests/js/live_games_test.js`; modify `web/static/css/style.css`, `web/templates/base.html`, `web/templates/games.html`, `web/templates/leaderboard.html`.

**Interfaces:** Consume the Task 1 snapshot and endpoint exactly. Public page endpoint is `live_games`; use `url_for('live_games')` in navigation.

- [x] Write a failing Node test using the existing fake-DOM/vm pattern: snapshots advance the board, navigating back survives a newer snapshot, Go live jumps to the latest move, selection changes between concurrent games, fetched names are rendered as text, and connection failures remain distinct from empty lists.
- [x] Run `node tests/js/live_games_test.js`, confirm it fails for the absent viewer, then implement the viewer and controls.
- [x] Keep API fetches sequential with a 4-second timer, no overlapping requests, hidden-tab pause and immediate refresh on visibility restoration. Validate payload shape before rendering.
- [x] Provide loading, no-live-games and connection-error copy, live/finished/interrupted status labels, a selected-game board, move ledger, flip board, start/previous/next/Go live buttons, and completed-game replay link.
- [x] Match existing styling and verify mobile layout and keyboard focus.
- [x] Add Live Games links to the shared navigation and Games/Leaderboard page navigation; preserve existing pages and DOM behavior.
- [x] Run the new Node tests and existing navigation tests.

## Integration and delivery

- [x] Review both task diffs for spec compliance and code quality, then review the integrated feature.
- [x] Run all focused Python/JS regressions and the repository suite; document pre-existing failures separately.
- [x] Start the local Flask app, load `/live`, and visually inspect a deterministic simulated live game. Verify updates, history navigation, empty/error states, mobile layout, and terminal replay transition without consuming LLM tokens.
- [x] Complete and verify the independently running Astra–Argon result publication; never replay a completed game to retry publication.
- [x] Commit only the requested feature and prior authorized model-registration changes, preserving unrelated data/output changes.
- [ ] Prepare a concrete reviewable delivery; shared-branch merge/production deployment follows the integration decision required by the applicable development skills.
