# Admin LLM Play Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add Firebase-backed Google login for all users and an interactive human-versus-LLM chess page restricted to `johnstondaniel4@gmail.com`.

**Architecture:** The browser signs in with Firebase Authentication and posts its ID token to Flask. Flask verifies the token with `firebase-admin`, stores only verified identity fields in its signed session, and derives admin status from a normalized `ADMIN_EMAILS` allowlist. Admin game state is a compact signed-session payload containing the selected configured model, human color, UCI move history, and terminal status; each move reconstructs the board, validates the human move server-side, obtains the configured LLM move, and returns authoritative state. Human games are never written to benchmark PGNs, ratings, or Firestore.

**Tech Stack:** Python 3.11, Flask 3, Firebase Admin SDK, Firebase Authentication Web SDK, python-chess, pytest, Jinja, vanilla JavaScript, Chessboard.js.

**Spec:** None — settled requirements are captured in Global Constraints and the tasks below.

## Global Constraints

- Any Firebase account with a verified email may establish a site session.
- Only normalized emails in `ADMIN_EMAILS` may access `/admin/*` or `/api/admin/*`; production configuration designates `johnstondaniel4@gmail.com`.
- Authentication and authorization fail closed when Firebase or required secrets are missing.
- State-changing requests require a per-session CSRF token sent in `X-CSRF-Token`.
- The server treats client model IDs, colors, moves, and game state as untrusted input.
- Only configured, available OpenRouter/completion/Gemini LLM entries may be selected; Codex CLI entries are excluded from the web deployment.
- Human games do not affect benchmark history, ratings, cache invalidation, or stability-cap data.
- Existing unrelated working-tree changes are preserved and no repository commit is created.

---

### Task 1: Firebase Session Authentication and Admin Authorization

**Files:**
- Create: `web/auth.py`
- Create: `web/templates/login.html`
- Create: `tests/test_web_auth.py`
- Modify: `web/app.py`
- Modify: `web/templates/base.html`

**Interfaces:**
- Produces: `verify_firebase_token(id_token: str) -> dict`, `current_user() -> dict | None`, `is_admin(user: dict | None = None) -> bool`, `login_required`, `admin_required`, `admin_api_required`, `validate_csrf() -> None`.
- Produces routes: `GET /login`, `POST /api/auth/session`, `POST /logout`.

- [ ] Write Flask-client tests proving anonymous users cannot reach the admin page/API, verified Firebase users can create sessions, an unverified email is rejected, the designated email is recognized as admin, non-admin accounts receive 403, invalid CSRF receives 403, and logout clears identity.
- [ ] Run `pytest tests/test_web_auth.py -v` and confirm failures are caused by missing authentication routes/helpers.
- [ ] Implement the minimal token verification, signed-session identity, CSRF, decorators, login template, and user-aware base navigation.
- [ ] Run `pytest tests/test_web_auth.py -v` and confirm all authentication/authorization tests pass.

### Task 2: Stateful Human-versus-LLM Game Service

**Files:**
- Create: `web/play_service.py`
- Create: `tests/test_web_play_service.py`

**Interfaces:**
- Produces: `list_playable_models(config_path: Path, environ: Mapping[str, str]) -> list[dict]`.
- Produces: `start_game(model_id: str, human_color: str, config_path: Path, environ: Mapping[str, str], move_provider=None) -> dict`.
- Produces: `play_human_move(state: dict, move_uci: str, config_path: Path, environ: Mapping[str, str], move_provider=None) -> dict`.
- State contains only `model_id`, `model_name`, `human_color`, `moves`, `status`, `winner`, and `termination`; response views additionally include FEN, SAN history, turn, and last move.

- [ ] Write service tests proving model allowlisting, unsupported backend filtering, valid start state, LLM opening move when the human chooses black, rejection of illegal/out-of-turn human moves without mutation, legal human+LLM turn progression, promotion support, game-over outcome reporting, LLM illegal-move retry/forfeit, and API-error state preservation.
- [ ] Run `pytest tests/test_web_play_service.py -v` and confirm failures are caused by the missing service.
- [ ] Implement board reconstruction, state validation, configured LLM creation, retry policy, response serialization, and cleanup without saving results.
- [ ] Run `pytest tests/test_web_play_service.py -v` and confirm all service tests pass.

### Task 3: Admin Gameplay API and Chessboard UI

**Files:**
- Create: `web/templates/play.html`
- Create: `web/static/js/play.js`
- Create: `tests/test_web_play_routes.py`
- Modify: `web/app.py`
- Modify: `web/static/css/style.css`

**Interfaces:**
- Consumes: `admin_required`, `admin_api_required`, `validate_csrf`, `list_playable_models`, `start_game`, and `play_human_move`.
- Produces routes: `GET /admin/play`, `POST /api/admin/play/start`, `POST /api/admin/play/move`.
- JSON success response: `{game: {fen, moves, san_moves, human_color, turn, status, winner, termination, last_move}}`.
- JSON error response: `{error: string}` with 400 for invalid input/state, 403 for authorization/CSRF, 503 for missing provider configuration, and 502 for upstream LLM failures.

- [ ] Write route tests proving admin-only access, CSRF enforcement, JSON schema, session persistence, invalid request errors, and provider failures that leave the prior game state intact.
- [ ] Run `pytest tests/test_web_play_routes.py -v` and confirm failures are caused by missing routes.
- [ ] Implement the routes and session update ordering so state changes only after successful service calls.
- [ ] Build a responsive two-column board/control room matching the existing navy/red visual system; use the move ledger as the single signature element, preserve visible keyboard focus, and expose status through an ARIA live region.
- [ ] Run `pytest tests/test_web_play_routes.py -v` and confirm all route tests pass.

### Task 4: Deployment Configuration, Documentation, and Verification

**Files:**
- Modify: `render.yaml`
- Modify: `README.md`

**Interfaces:**
- Requires Render secrets: `FLASK_SECRET_KEY`, `FIREBASE_WEB_API_KEY`, `ADMIN_EMAILS`, and at least one of `OPENROUTER_API_KEY` or `GEMINI_API_KEY`.
- Uses Firebase project ID/auth domain derived from server credentials unless explicitly overridden by `FIREBASE_PROJECT_ID`/`FIREBASE_AUTH_DOMAIN`.

- [ ] Add Render secret declarations without values, designate `johnstondaniel4@gmail.com` through `ADMIN_EMAILS`, and set a Gunicorn timeout compatible with LLM turns.
- [ ] Document Firebase Google-provider/authorized-domain setup and required Render environment variables.
- [ ] Run `pytest tests/test_web_auth.py tests/test_web_play_service.py tests/test_web_play_routes.py -v`.
- [ ] Run the complete repository test suite with `pytest -q` and record any unrelated pre-existing failures separately.
- [ ] Start the Flask app with test environment values, inspect login and play pages at desktop/mobile widths, verify browser console/network behavior, and remove any visual or accessibility regressions.
- [ ] Review the final diff for leaked credentials, accidental benchmark writes, authorization gaps, and overlap with pre-existing user changes.
