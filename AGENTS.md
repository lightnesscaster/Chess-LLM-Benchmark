# Repository Guidelines

## Project Structure & Module Organization
- `cli.py` is the main entrypoint for running benchmarks, leaderboards, and manual games.
- `config/benchmark.yaml` holds matchup, engine, and LLM settings; edit here for most changes.
- Engines live in `engines/` (Stockfish, Maia, random, UCI, survival presets), LLM clients in `llm/`, and core game orchestration in `game/` (match scheduling, PGN logging, stats).
- Rating logic (Glicko-2, cost, leaderboard formatting) is in `rating/`.
- Web UI is under `web/` (Flask app, templates, static assets). Generated game/result data is stored in `data/` (gitignored).
- Utility scripts: `scripts/fetch_pricing.py`, `cleanup_bad_games.py`, `migrate_to_firestore.py`, etc.

## Build, Test, and Development Commands
- Install deps: `pip install -r requirements.txt`
- Run full benchmark: `python cli.py run -c config/benchmark.yaml -v`
- Manual game (good smoke test): `python cli.py manual --white-model meta-llama/llama-4-maverick --black-engine`
- Leaderboard render: `python cli.py leaderboard --min-games 5`
- Recompute ratings after new games: `python cli.py recalculate -c config/benchmark.yaml`
- Web app (local): `python web/app.py` then open `http://localhost:5000`

## Coding Style & Naming Conventions
- Python, 4-space indents, type hints throughout; follow existing pattern of concise functions and explicit error messages.
- Use snake_case for functions/vars, PascalCase for classes, UPPER_SNAKE_CASE for constants.
- Prefer explicit imports from local modules; keep configuration and magic numbers in `config/benchmark.yaml` when possible.
- Match current formatting (double quotes in this codebase) and add docstrings for CLI-facing functions.

## Testing Guidelines
- No formal test suite yet; run a manual smoke (`cli.py manual`) plus a short benchmark subset to validate changes touching game flow, engines, or rating logic.
- For web changes, start `python web/app.py` and load leaderboard/game pages to confirm templates and JSON endpoints.
- If adding tests, colocate them near the module or in a `tests/` directory; name files `test_<module>.py`.

## Commit & Pull Request Guidelines
- Commit messages are short and prefixed when relevant (`feat:`, `refactor:`, etc.) with an imperative summary (see `git log`).
- PRs should include: what changed, why, how to reproduce (commands/config snippets), and any output screenshots for web/UI updates.
- Reference related issues or configs updated (e.g., `config/benchmark.yaml`), and note any new environment variables or data expectations.

## Security & Configuration Tips
- Keep secrets out of the repo; set `OPENROUTER_API_KEY` (and optional `WEB_APP_URL`/`CACHE_INVALIDATE_TOKEN`) via environment variables.
- Avoid committing `data/` outputs or private keys (e.g., Firebase); ensure added configs do not contain credentials.
