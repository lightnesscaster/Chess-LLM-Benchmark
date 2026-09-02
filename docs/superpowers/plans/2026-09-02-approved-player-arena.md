# Approved Player Arena Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let Firestore-approved Google accounts play rated games against available LLMs using an automatically fetched Lichess Rapid rating and RD snapshot.

**Architecture:** Store approved player records in Firestore keyed by normalized email, with admins implicitly approved. At game start, resolve the player's saved Lichess username through the public Lichess API and place the rating/RD snapshot in signed session state. When a game finishes, persist an idempotent human-challenge result and PGN, update only the LLM through Glicko-2, and teach full recalculation to replay the immutable human snapshot as a fixed opponent.

**Tech Stack:** Flask, Firebase Admin/Firestore, requests, python-chess, Pydantic, Glicko-2, vanilla JavaScript/CSS, pytest, Node-based browser-script tests.

**Spec:** None — settled requirements are captured in Global Constraints and the tasks below.

## Global Constraints

- Only verified Google accounts in the Firestore approval list may access the arena; configured admins are implicitly approved.
- Admins can add and remove approved email addresses from the website without a deployment.
- Players enter a Lichess username; the server fetches Rapid rating and RD at every game start.
- The immutable game-start snapshot, including provisional status, is stored with the completed game.
- Completed games update only the selected LLM; the human is a fixed one-game opponent.
- Missing Rapid data, unavailable Lichess profiles, unfinished games, and duplicate finish requests never affect ratings.
- Existing benchmark results and admin access continue to work.
- The arena keeps its existing chess-club visual system and remains responsive and keyboard accessible.

---

### Task 1: Approved-player authorization and administration

**Files:**
- Create: `web/approved_players.py`
- Create: `web/templates/approved_players.html`
- Modify: `web/auth.py`
- Modify: `web/app.py`
- Modify: `web/templates/base.html`
- Modify: `web/static/css/style.css`
- Test: `tests/test_web_auth.py`
- Test: `tests/test_web_approved_players.py`

**Interfaces:**
- Produces: `ApprovedPlayerStore`, `is_approved_player(user)`, `player_required`, and `player_api_required`.
- Produces: admin routes `GET /admin/players`, `POST /api/admin/players`, and `DELETE /api/admin/players/<email>`.

- [ ] Write failing authorization and admin-management route tests, including normalized emails, anonymous access, non-approved access, admin implicit access, duplicate approval, and removal.
- [ ] Run the focused tests and verify failures are caused by missing approved-player behavior.
- [ ] Implement Firestore/local-test-compatible approval storage, decorators, routes, page, navigation, and accessible controls.
- [ ] Run the focused tests and verify they pass.

### Task 2: Lichess Rapid profile snapshots

**Files:**
- Create: `web/lichess.py`
- Modify: `web/play_service.py`
- Test: `tests/test_web_lichess.py`
- Test: `tests/test_web_play_service.py`

**Interfaces:**
- Produces: `LichessRapidSnapshot(username, rating, rating_deviation, provisional)` and `fetch_rapid_snapshot(username)`.
- Extends: `start_game(..., human_profile=...)` and the signed game state/view with the immutable snapshot, UUID, and start timestamp.

- [ ] Write failing tests for valid profile parsing, canonical username use, missing Rapid data, malformed responses, HTTP failures, snapshot validation, and game-state serialization.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement the Lichess client with explicit timeout/user-agent and state snapshot validation.
- [ ] Run the focused tests and verify they pass.

### Task 3: Persist and score completed human challenges

**Files:**
- Modify: `game/models.py`
- Create: `web/human_challenges.py`
- Modify: `firebase_client.py`
- Modify: `web/app.py`
- Modify: `rating/rating_store.py`
- Test: `tests/test_human_challenges.py`
- Test: `tests/test_web_play_routes.py`

**Interfaces:**
- Extends: `GameResult` with backward-compatible `game_type` and human snapshot fields.
- Produces: `build_human_challenge_result(state, email)` and `record_human_challenge(state, email)`.
- Persists: PGN, result, and updated LLM rating once per game ID; returns the resulting model rating summary.

- [ ] Write failing pure scoring/result tests and route tests proving only completed games score, human RD changes result weight, and a duplicate record is idempotent.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement result construction, PGN headers, transactional Firestore persistence with local fallback, one-sided Glicko update, and cache invalidation.
- [ ] Run the focused tests and verify they pass.

### Task 4: Deterministic recalculation with fixed human snapshots

**Files:**
- Modify: `cli.py`
- Modify: `game/stats_collector.py`
- Test: `tests/test_rating_recalculation.py`
- Test: `tests/test_human_challenges.py`

**Interfaces:**
- Consumes: human-challenge `GameResult` snapshot fields.
- Produces: recalculated model ratings that replay human opponents at stored rating/RD without creating leaderboard entries for humans.

- [ ] Write failing recalculation tests proving the human is excluded as a rated player and changing stored human RD changes the model's update magnitude.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Extend rating replay and statistics collection for one-sided human-challenge games while leaving benchmark replay unchanged.
- [ ] Run the focused tests and verify they pass.

### Task 5: Player arena identity and rating UI

**Files:**
- Modify: `web/app.py`
- Modify: `web/templates/play.html`
- Modify: `web/static/js/play.js`
- Modify: `web/static/css/style.css`
- Modify: `tests/test_web_play_routes.py`
- Modify: `tests/test_web_play_client.py`
- Modify: `tests/js/play_effort_selector_test.js`

**Interfaces:**
- Generalizes: `/admin/play` to canonical `/play` while preserving the old URL as a redirect.
- Adds: `lichess_username` to start-game requests and rating-snapshot/model-rating feedback to responses.

- [ ] Write failing route and client tests for approved access, username submission, visible Rapid snapshot, rated-result messaging, and existing export/effort behavior.
- [ ] Run the focused tests and verify the expected failures.
- [ ] Implement the player-facing setup strip, approval-aware navigation, start payload, and scored-game status copy without diluting the board-first layout.
- [ ] Run Python and JavaScript focused tests and verify they pass.

### Task 6: Full verification, commit, push, and deployment check

**Files:**
- Modify: `README.md` or deployment documentation only if a new operational requirement is introduced.

- [ ] Run the complete Python and JavaScript test suites.
- [ ] Run a local Flask smoke test for anonymous, approved, and admin routes.
- [ ] Inspect the diff for secrets and unrelated changes.
- [ ] Commit on `main` with a concise `feat:` message and push using the configured `lightnesscaster` credentials.
- [ ] Verify the Render deployment reaches live status and smoke-test the production authorization boundary.
